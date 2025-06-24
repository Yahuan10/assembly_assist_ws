import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/rick/Documents/MyProject/assembly_assist_ws/install/fake_camera_node'
