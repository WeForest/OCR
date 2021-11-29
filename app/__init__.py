# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
#from werkzeug import secure_filename
from PIL import Image
import os
import cv2
import json
import glob
import numpy as np
import itertools
from itertools import chain

from pdf2image import convert_from_path

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from pre_dataset import RawDataset, AlignCollate
from model import Model
from util import AttnLabelConverter

import argparse
import shutil
import time
from pathlib import Path
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

import torch.nn as nn
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert.pytorch_kobert import get_pytorch_kobert_model
import numpy as np
from torch.utils.data import Dataset, DataLoader
from kobert.utils import get_tokenizer
import gluonnlp as nlp

max_len = 100
batch_size = 64
warmup_ratio = 0.1
num_epochs = 4
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


### GPU Setting  : Not Using GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

app = Flask(__name__)

# 이미지 업로드
@app.route('/fileUpload', methods=['POST'])
def upload_file():
    data = Image.open(request.files['image'])   # data로 된 데이터 가져옴
    #data = cv2.imread(request.files['image'], cv2.IMREAD_UNCHANGED)
    data.save('./img/data.png', 'png')
    #cv2.imwrite('data.png', './img/')
    
    target = ['./img/data.png']
    data = run(target)
    print(data)

    d = {'success' : True, 'conference' : 'conferenceName', 'name':'idonknow'}
    return jsonify(d)

# 혐오, 욕설 검출
@app.route('/abuse', methods=['GET'])
def abuse():
    data = request.args['data']   # data로 된 데이터 가져옴
    a= predict(data)
    #if data == 0 :
    #    return None
    #sprint('end')
    return str(a)


class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()
        self.num_class = num_class
        self.stages = {'Trans': True, 'Feat': True,
                       'Seq': True, 'Pred': True}

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=20, I_size=(32, 100), I_r_size=(32, 100), I_channel_num=1)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(1, 512)
        self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, 256, 256),
            BidirectionalLSTM(256, 256, 256))
        self.SequenceModeling_output = 256
        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, 256, num_class)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction


def detect(save_img=False):
    save_dir = './yolo'
    my_source = './img'
    my_weights = './weight/detect/best.pt'
    my_img_size = 640
    conf_thres = 0.25
    iou_thres = 0.01
    if torch.cuda.is_available():
        my_device= '0'
    else:
        my_device = 'cpu'
    out, source, weights, view_img, save_txt, imgsz = \
        save_dir, my_source, my_weights, False, True, my_img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    
    # Initialize
    set_logging()
    device = select_device(my_device)
    if os.path.exists(out):  # output dir
        shutil.rmtree('./yolo', ignore_errors=True)  # delete dir

    #os.makedirs(out)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    imgsz = check_img_size(imgsz, s=detect_model.stride.max())  # check img_size
    if half:
        detect_model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = detect_model.module.names if hasattr(detect_model, 'module') else detect_model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = detect_model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = detect_model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh)   # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        #label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label="", color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

def convertCoordination():
    my_loc=[]
    len_loc = []
    txt_path = r'./yolo/'
    txt_dir = glob.glob(txt_path + '/*.txt')
    img_path = r'./img/'
    img_dir = glob.glob(img_path + '/*.jpg')
    for _img,_txt in zip(img_dir, txt_dir):
        #빈 리스트 생성
        coordinate = []
        #dh dw : 이미지의 가로, 세로 크기
        img = cv2.imread(_img)
        dh, dw, _ = img.shape
        #YOLO 좌표가 있는 텍스트 파일 불러옴
        fl = open(_txt, 'r')
        data = [line[:-1] for line in fl.readlines()]
        fl.close()
        for dt in data:
            # Split string to float
            _, x, y, w, h = map(float, dt.split())
            # l : left , r : right , t : top, b : bottom
            # x = l , y = t , w = r-l, h = b-t
            l = int((x - w / 2) * dw)
            r = int((x + w / 2) * dw)
            t = int((y - h / 2) * dh)
            b = int((y + h / 2) * dh)
            # 좌표 후처리(없어도 무관)
            # left 값이 - 일 경우 0으로 저장
            if l < 0:
                l = 0
            # right 값이 (전체 이미지 사이즈) - 1 보다 클 경우
            if r > dw - 1:
                r = dw - 1
            # top 값이 - 일 경우 0으로 저장
            if t < 0:
                t = 0
            # bottom 값이 (전체 이미지 사이즈) - 1 보다 클 경우
            if b > dh - 1:
                b = dh - 1
            #리스트에 저장
            coordinate.append([l,t,r-l,b-t])
        coordinate.sort(key=lambda x : x[1])
        
        s = []
        loc = []
        for i in range(1 , len(coordinate)):
            if abs( coordinate[i-1][1] - coordinate[i][1] ) < 10:
                s.append(coordinate[i-1])
            else:
                if len(s) == 0:
                    loc.append([coordinate[i-1]])
                else:
                    s.append(coordinate[i-1])
                    s.sort(key=lambda x :x[0])
                    loc.append(s)
                    s = []
                    
        s.append(coordinate[i-1])
        s.sort(key=lambda x :x[0])
        loc.append(s)

        len_loc.append(list(map(lambda x :len(x),loc)))
        my_loc.append(list(itertools.chain(*loc)))
        
    return len_loc,my_loc

#만들어낸 loc 리스트를 불러와 좌표값에 따른 이미지 추출하여 저장
def read_img_by_coord (loc):
    img_list=[]
    img_path = r'./img/'
    img_dir = glob.glob(img_path + '/*.jpg')
    for _img,_loc in zip(img_dir, loc):
        print("make image", os.path.basename(_img[:-4]))
        org_image = cv2.imread(_img,cv2.IMREAD_GRAYSCALE)
        cv_list = []
        for i in _loc:
            img_trim = org_image[i[1]:i[1]+i[3], i[0]:i[0]+i[2]] #trim한 결과를 img_trim에 담는다
            #dst = cv2.bitwise_not(img_trim)
            cv_list.append(img_trim)
        img_list.append(cv_list)
    return img_list



def run(target):
    # target = ["./test/10-20190207162430645.jpg"]
    # target = ["D:/OCR_MAIN/test/10-20190207162430645.jpg","D:/OCR_MAIN/test/10-20190207170104526.jpg"]

    SAVE_JSON = False

    img = []
    predict = []
    data = []
    img_dir = './img'

    for i in target:
        ext = os.path.splitext(i)[-1]
        if ext.lower() == '.pdf':
            pages = convert_from_path(i, dpi=200, poppler_path='./poppler-0.68.0/bin')
            for j, page in enumerate(pages):
                page.save(f'{img_dir}/{os.path.basename(i)[:-4]}_page{j + 1:0>2d}.jpg')
        elif ext.lower() == '.jpg' or ext.lower() == '.png':
            img = cv2.imread(i, 0)
            print(img)
            cv2.imwrite('{}/{}.jpg'.format(img_dir, os.path.basename(i)[:-4]), img)
        elif ext.lower() == '.tif':
            img = cv2.imread(i, 0)
            cv2.imwrite('{}/{}.jpg'.format(img_dir, os.path.basename(i)[:-4]), img)

    result = [[] for _ in range(len(target))]

    # time : 2.855405569076538
    #detect
    img = glob.glob('./img/*.jpg')

    #for im in img:
    #    pre_img = pr.skewCorrection(im)
    #    cv2.imwrite('./img/pre_{}'.format(os.path.basename(im)), pre_img)
    #    os.remove(im)
    with torch.no_grad():
        detect()

    len_loc, loc = convertCoordination()

    for i in range(len(len_loc)):
        for j in range(1, len(len_loc[i])):
            len_loc[i][j] = len_loc[i][j] + len_loc[i][j - 1]

    detect_output = read_img_by_coord(loc)

    for i in range(len(detect_output)):
        for j, img in enumerate(detect_output[i]):
            cv2.imwrite("./temp/{}.jpg".format(j), img)

        images = ["./temp/{}.jpg".format(x) for x in range(len(detect_output[i]))]


    AlignCollate_demo = AlignCollate(imgH=32, imgW=100)
    demo_data = RawDataset(root='./temp')  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=128,
        shuffle=False,
        num_workers=int(0),
        collate_fn=AlignCollate_demo, pin_memory=True)

    recognize_model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([25] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).to(device)

            preds = recognize_model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                if '\\"' in pred:
                    pred = pred.replace('\\"','\"')

                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                data.append(pred)
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
    print(data)
    for file in os.listdir('./img/'):
        os.remove("./img/{}".format(file))
    for k in glob.glob('./temp/*.jpg'):
        os.remove(k)
    return data

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]
        #self.labels = [j for i, j in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

def predict(predict_sentence):
    print('predict 호출')

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    nlp_model.eval()
    print('model eval 실행')
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        print('check1')
        valid_length= valid_length
        label = label.long().to(device)
        print('check2')
        out = nlp_model(token_ids, valid_length, segment_ids)
        print()

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append(False)
            elif np.argmax(logits) == 1:
                test_eval.append(True)

        print(test_eval[0])
        return test_eval[0] 


if __name__ == '__main__':
    global nlp_model
    global detect_model
    global recognize_model
    global tokenizer
    global tok
    

    device = torch.device("cpu")
    bertmodel, vocab = get_pytorch_kobert_model()
    nlp_model = BERTClassifier(bertmodel,  dr_rate=0.5)
    nlp_model.load_state_dict(torch.load('./weight/nlp/model.pt', map_location=device))

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


    #recognition
    char_list = '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝!@#$%^&*《》()[]【】【】\"\'◐◑oㅇ⊙○◎◉◀▶⇒◆■□△★※☎☏;:/.?<>-_=+×\￦|₩~,.㎡㎥ℓ㎖㎘→「」『』·ㆍ1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ읩①②③④⑤月日軍 '

    converter = AttnLabelConverter(char_list)
    num_class = len(converter.character)
    
    recognize_model = Model(num_class)
    recognize_model = torch.nn.DataParallel(recognize_model).to(device)
    print('loading pretrained model from %s' % './weight/recognize/best_accuracy.pth')
    recognize_model.load_state_dict(torch.load('./weight/recognize/best_accuracy.pth', map_location=device))
    print('load ending')

    # Load model
    weights = './weight/detect/best.pt'
    detect_model = attempt_load(weights, map_location=device)  # load FP32 model
    print('server run')
    app.run(host='0.0.0.0', port=5000, debug=True)

