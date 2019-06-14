from AiCampEval import eval_submit


def dumb(nparray):
    leng = len(nparray)
    return ['ChairPose']*leng

eval_submit(dumb, 'testset_11classes_1_01010', 'PlayerA')