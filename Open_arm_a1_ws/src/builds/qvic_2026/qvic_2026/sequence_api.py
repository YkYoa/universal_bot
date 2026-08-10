"""Flask blueprint for the sequence store: the teaching-pendant CRUD.

Registered by moveit_api's robot_api_server if this package is importable, and
skipped if it is not. The dependency runs one way on purpose - the generic API
server knows nothing about this project's store, so it still works in a
workspace that does not have it.

Everything here is store manipulation. Actually running a sequence goes through
the FSM endpoints in robot_api_server, because that is a ROS call, not a
database one.
"""

from flask import Blueprint, jsonify, request

from . import step_types, store

bp = Blueprint('qvic_sequences', __name__)

# Set by register(); used by POST /api/waypoints to capture the arm's current
# position. None means "no ROS available", and that endpoint says so.
_joint_state_reader = None


def register(app, joint_state_reader=None):
    """Attach the blueprint to `app`.

    `joint_state_reader` is an optional callable returning
    {joint_name: position} - robot_api_server passes its MoveIt controller's
    live joint state so a waypoint can be recorded from the browser the same
    way `record_waypoint` does from the terminal.
    """
    global _joint_state_reader
    _joint_state_reader = joint_state_reader
    app.register_blueprint(bp)
    return bp


def _fail(message, code=400):
    return jsonify({'success': False, 'message': message}), code


def _body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError('a JSON object body is required')
    return data


def _handle(fn):
    """Store errors are user errors - a missing sequence, a bad step - so they
    are 400s with the store's own message, not 500s with a stack trace."""
    try:
        return fn()
    except store.StoreError as exc:
        return _fail(str(exc))
    except ValueError as exc:
        return _fail(str(exc))


# ── catalog ──────────────────────────────────────────────────────────────────

@bp.route('/api/step-types', methods=['GET'])
def get_step_types():
    """The drag-and-drop palette. The Android app builds its step editor from
    this rather than hardcoding a list that would drift from the executor."""
    return jsonify({
        'success': True,
        'step_types': step_types.catalog(),
        'control_modes': {
            'any': 'Runs in any hardware control mode.',
            'position|mit': 'Needs the arm commanding positions (position or mit).',
            'torque': 'Needs the arm compliant (torque), i.e. gravity compensation.',
        },
        'arms': list(step_types.ARMS),
        'sides': list(step_types.SIDES),
    })


# ── sequences ────────────────────────────────────────────────────────────────

@bp.route('/api/sequences', methods=['GET'])
def list_sequences():
    return jsonify({'success': True, 'sequences': store.list_sequences()})


@bp.route('/api/sequences', methods=['POST'])
def create_sequence():
    def run():
        data = _body()
        name = data.get('name')
        if not name:
            return _fail("'name' is required")
        seq = store.create_sequence(
            name,
            description=data.get('description', ''),
            arm=data.get('arm', 'left_arm'),
            planner_profile=data.get('planner_profile', ''),
            repeat=data.get('repeat', 1),
            velocity=data.get('velocity', 0.0),
            acceleration=data.get('acceleration', 0.0),
            steps=data.get('steps'),
        )
        return jsonify({'success': True, 'sequence': seq}), 201
    return _handle(run)


@bp.route('/api/sequences/<name>', methods=['GET'])
def get_sequence(name):
    return _handle(lambda: jsonify({'success': True, 'sequence': store.get_sequence(name)}))


@bp.route('/api/sequences/<name>', methods=['PUT'])
def update_sequence(name):
    def run():
        data = _body()
        fields = {k: v for k, v in data.items() if k in store.UPDATABLE_FIELDS}
        seq = store.update_sequence(name, **fields)
        if data.get('steps') is not None:
            seq = store.replace_steps(name, data['steps'])
        if data.get('new_name') and data['new_name'] != name:
            seq = store.rename_sequence(name, data['new_name'])
        return jsonify({'success': True, 'sequence': seq})
    return _handle(run)


@bp.route('/api/sequences/<name>', methods=['DELETE'])
def delete_sequence(name):
    def run():
        store.delete_sequence(name)
        return jsonify({'success': True, 'message': f"deleted '{name}'"})
    return _handle(run)


@bp.route('/api/sequences/<name>/duplicate', methods=['POST'])
def duplicate_sequence(name):
    def run():
        data = _body()
        new_name = data.get('new_name') or f'{name}_copy'
        return jsonify({'success': True, 'sequence': store.duplicate_sequence(name, new_name)}), 201
    return _handle(run)


# ── steps ────────────────────────────────────────────────────────────────────

@bp.route('/api/sequences/<name>/steps', methods=['POST'])
def add_step(name):
    def run():
        data = _body()
        index = data.pop('index', None)
        return jsonify({'success': True, 'sequence': store.add_step(name, data, index=index)}), 201
    return _handle(run)


@bp.route('/api/sequences/<name>/steps/<int:index>', methods=['PUT'])
def update_step(name, index):
    return _handle(
        lambda: jsonify({'success': True, 'sequence': store.update_step(name, index, _body())})
    )


@bp.route('/api/sequences/<name>/steps/<int:index>', methods=['DELETE'])
def delete_step(name, index):
    return _handle(
        lambda: jsonify({'success': True, 'sequence': store.delete_step(name, index)})
    )


@bp.route('/api/sequences/<name>/reorder', methods=['POST'])
def reorder_steps(name):
    def run():
        order = _body().get('order')
        if not isinstance(order, list):
            return _fail("'order' must be a list of the current step indices in their new order")
        return jsonify({'success': True, 'sequence': store.reorder_steps(name, order)})
    return _handle(run)


# ── waypoints ────────────────────────────────────────────────────────────────

@bp.route('/api/waypoints', methods=['GET'])
def list_waypoints():
    return jsonify({
        'success': True,
        'waypoints': store.list_waypoints(section=request.args.get('section')),
        'sections': store.list_sections(),
    })


@bp.route('/api/waypoints', methods=['POST'])
def create_waypoint():
    """Record a waypoint, either from explicit values or from where the arm is
    right now (`source: "live"`) - the browser equivalent of `record_waypoint`."""
    def run():
        data = _body()
        name = data.get('name')
        section = data.get('section')
        if not name or not section:
            return _fail("'name' and 'section' are required")

        values = data.get('values')
        if data.get('source') == 'live':
            if _joint_state_reader is None:
                return _fail('live capture needs the ROS bridge, which is not available', 503)
            arm = data.get('arm', 'left_arm')
            values = _joint_state_reader(arm)
            if not values:
                return _fail(f"no joint state for '{arm}' yet - is the robot publishing?", 503)
        if not values:
            return _fail("'values' is required unless source is 'live'")

        waypoint = store.upsert_waypoint(
            name=name,
            section=section,
            arm_prefix=data.get('arm_prefix', name[:2]),
            kind=data.get('kind', 'angle'),
            values=values,
        )
        return jsonify({'success': True, 'waypoint': waypoint}), 201
    return _handle(run)


@bp.route('/api/waypoints/<path:ref>', methods=['GET'])
def get_waypoint(ref):
    # `ref` is 'section/name'; the path converter keeps the slash.
    return _handle(lambda: jsonify({'success': True, 'waypoint': store.get_waypoint(ref)}))


@bp.route('/api/waypoints/<path:ref>', methods=['DELETE'])
def delete_waypoint(ref):
    def run():
        store.delete_waypoint(ref)
        return jsonify({'success': True, 'message': f"deleted '{ref}'"})
    return _handle(run)


# ── YAML sync ────────────────────────────────────────────────────────────────

@bp.route('/api/store/import', methods=['POST'])
def import_yaml():
    def run():
        from . import yaml_sync
        path = _body().get('file')
        if not path:
            return _fail("'file' is required")
        return jsonify({'success': True, 'summary': yaml_sync.import_yaml(path)})
    return _handle(run)


@bp.route('/api/store/export', methods=['POST'])
def export_yaml():
    def run():
        from . import yaml_sync
        path = _body().get('file')
        if not path:
            return _fail("'file' is required")
        return jsonify({'success': True, 'summary': yaml_sync.export_yaml(path)})
    return _handle(run)


@bp.route('/api/store/runs', methods=['GET'])
def list_runs():
    limit = request.args.get('limit', 50)
    return _handle(lambda: jsonify({'success': True, 'runs': store.list_runs(limit=limit)}))


@bp.route('/api/store/info', methods=['GET'])
def store_info():
    return jsonify({
        'success': True,
        'db_path': store.db_path(),
        'sequences': len(store.list_sequences()),
        'waypoints': len(store.list_waypoints()),
    })
