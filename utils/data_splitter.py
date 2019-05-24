import os

overall_dir = "/media/dh/Data/AI_Camp/"
binary_class = "WarriorPose"
set_name = "train"
# set_name = "test"
source_name="TIL2019_v0.1/{}".format(set_name)
target_name="binaries/{}/{}".format(binary_class, set_name)
mapping = "mapping.cfg"

def mapping_parse(fp):
    with open(fp,'r') as f:
        lines = f.readlines()
    mapping = {}
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        if line.startswith('['):
            _class = line.replace('[','').replace(']','')
            mapping[_class] = []
        else:
            mapping[_class].append(line)
    return mapping

source_dir = os.path.join(overall_dir, source_name)
assert os.path.isdir(source_dir)
target_dir = os.path.join(overall_dir, target_name)
if not os.path.isdir(target_dir):
    os.makedirs(target_dir)
assert os.path.isdir(target_dir)

assert os.path.exists(mapping)
mapping = mapping_parse(mapping)

for target_class in mapping:
    print('\nDoing {}'.format(target_class))
    target_class_dir = os.path.join( target_dir, target_class)
    if not os.path.isdir(target_class_dir):
        os.makedirs(target_class_dir)
    assert os.path.isdir(target_class_dir)

    for source_class in mapping[target_class]:
        print(source_class)
        source_class_dir = os.path.join(source_dir, source_class)
        assert os.path.isdir(source_class_dir)

        for image_name in os.listdir(source_class_dir):
            dst = os.path.join(target_class_dir, image_name)
            if os.path.exists(dst):
                continue
            src = os.path.join(source_class_dir, image_name)
            # print('src:',src)
            # print('dst:',dst)
            src_relative = os.path.relpath(src, start=target_class_dir)
            # print(src_relative)
            os.symlink(src_relative, dst)
