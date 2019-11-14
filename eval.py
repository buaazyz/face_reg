from PIL import Image
from config import get_config
from train import face_learner
from utils import load_facebank, prepare_facebank
from tqdm import tqdm


def calac(labellist, predict):
    acc = 0
    for i in range(len(labellist)):
        a = labellist[i]
        b = predict[i]
        if a == b:
            acc += 1
    print(labellist)
    print(predict)
    print('per acc: ', end='')
    print(acc / len(labellist))
    return acc, len(labellist)


def caltp(labellist, predict):
    tp = tn = fp = fn = 0
    print(labellist)
    print(predict)
    for i in range(len(labellist)):
        a = labellist[i]
        b = predict[i]
        if a > -1 and a == b:
            tp += 1
        elif a > -1 and a != b:
            fp += 1
        elif a == -1 and a == b:
            tn += 1
        elif a == -1 and a != b:
            fn += 1
    print(tp, tn, fp, fn)
    return tp, tn, fp, fn


if __name__ == '__main__':

    conf = get_config(False)

    update = False

    learner = face_learner(conf, True)

    if conf.device.type == 'cpu':
        learner.load_state(conf, '2019-11-11-08-11.pth', True, True)
    else:
        learner.load_state(conf, 'model_2019-11-11-08-11.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    if update:
        targets, names = prepare_facebank(conf, learner.model)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
    imgs = []
    names = []

    testlist = []
    learner.threshold = 0.25
    with open(conf.testlist_path, 'r') as f:
        testlist = f.readlines()
        f.close()
    # print(testlist)
    acc = 0
    total = 0
    tp0 = tn0 = fp0 = fn0 = 0
    category = 'face94'

    neg_list = [224, 347, 356, 153, 26, 49, 217, 67, 124, 279, 36, 207, 196, 309, 20, 154, 390, 202, 118,
                            308, 97, 87, 183, 368, 237, 233, 21, 114, 388, 200, 295, 349, 213, 60, 230]

    facebanklist = list(range(394))
    for i in neg_list:
        facebanklist.remove(i)

    for k in tqdm(range(68)):
        templist = testlist[k * 20:(k + 1) * 20]
        imglist = []
        labellist = []
        for i in templist:
            item = i.split(',')
            img = item[0]
            label = item[1]
            cate = item[2]
            # print(cate)
            if cate[:-1] == category:
                label = int(label)
                labellist.append(label)
                path = conf.test_path / img
                data = Image.open(path)
                data = data.convert('RGB')
                imglist.append(data)
        if len(imglist) != 0:
            res, score = learner.infer(conf, imglist, targets, False)
            # acc1 ,total1 = calac(labellist, results)
            results = []
            for r in res:
                if r == -1:
                    results.append(-1)
                else:
                    results.append(facebanklist[r])
            tp, tn, fp, fn = caltp(labellist, results)
            # print(score)
            tp0 += tp
            tn0 += tn
            fp0 += fp
            fn0 += fn
            # acc += acc1
            # total += total1

    tpr = tp0 / (tp0 + fn0)
    fpr = fp0 / (tn0 + fp0)
    print('TPR:', end='')
    print(tpr)
    print('FPR:', end='')
    print(fpr)

    #         对于ROC来说，横坐标就是FPR，而纵坐标就是TPR

    # print(acc / total)

    # for idx, bbox in enumerate(bboxes):
    #     if args.score:
    #         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
    #     else:
    #         frame = draw_box_name(bbox, names[results[idx] + 1], frame)
