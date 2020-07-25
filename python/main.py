import dlib, cv2
import time
import numpy as np
import pandas as pd
import csv
import datetime
import os
from guizero import App, Text, TextBox

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('modules/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('modules/dlib_face_recognition_resnet_model_v1.dat')
INPUT_IMG = './input/input.jpg'
DATA_FILE_NAME = '/home/pi/python/customers.csv'

DATA = []


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_csv():

    f = pd.read_csv(DATA_FILE_NAME)

    df = {
        'face': [],
        'phone_number': None,
        'gender': None,
        'latest_visit_date': None,
        'latest_order': None,
        'recent_visits': None
    }

    for index, data in f.iterrows():
        df = set_df(data["face"], data["phone_number"], data["gender"],
                data["latest_visit_date"], data["latest_order"],
                data["recent_visits"])
        DATA.append(df)
    

def save_csv():
    pd.DataFrame(DATA).to_csv(DATA_FILE_NAME)

def find_faces(img):
    # 얼굴 찾기
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype = np.int)

    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
    
    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    # 얼굴 특징 추출
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))
    
    return np.array(face_descriptors)

def find_customer(descriptors, frame):
    
    for i, desc in enumerate(descriptors):
        found = False
        
        for index, row in enumerate(DATA):
            face = row["face"][0]['data']
         
            dist = np.linalg.norm([desc] - face, axis=1)
            print(dist)
            if dist < 0.5:
                print_info(row, frame, index)
                DATA[index]['recent_visits'] += 1 
                DATA[index]['latest_visit_date'] = datetime.datetime.now() 
                found=True
                break
        
        if not found:
            print("first visit")
            add_customer(descriptors)

    #return found


def check_image(frame):

    # 캡쳐한 이미지 불러오기
    # img_bgr = cv2.imread(INPUT_IMG)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 캡쳐한 이미지에 사람얼굴 찾기
    rects, shapes, _ = find_faces(img_rgb)

    # 발견한 사람 얼굴의 특징 추출
    descriptors = encode_faces(img_rgb, shapes)

    # 등록된 손님인지 확인

    find_customer(descriptors,img_rgb)

    


def check():
    print(DATA["face"])


def set_df(face, phone_number, gender, latest_visit_date, latest_order, recent_visits):
    data = face.replace(' ','')
    data = data[17:-19].replace('\n','') 
    data = np.array(data.split(','), dtype=np.float32)
    data = np.array([{'data': data}])

    df = {
    'face': data,
    'phone_number': phone_number,
    'gender': gender,
    'latest_visit_date': latest_visit_date,
    'latest_order': latest_order,
    'recent_visits': recent_visits
    }

    return df

def add_customer(descriptors):
    app = App(title="Add Customer")
    message = Text(app, text="First visit") 
    
    df = {
    'face': [],
    'phone_number': None,
    'gender': None,
    'latest_visit_date': None,
    'latest_order': None,
    'recent_visits': None
    }
    phoneM = Text(app, text="phone number") 
    phone = TextBox(app)
    

    genderM = Text(app, text = "W?M?")
    gender = TextBox(app)
    orderM = Text(app, text = "order") 
    latest_order = TextBox(app)
    app.display()
    feature = np.array([{'data':np.array(descriptors[0], dtype=np.float32)}])
    df["face"] = feature
    df["phone_number"] =phone.value 
    df["gender"] =gender.value 
    df["latest_visit_date"] = datetime.datetime.now()
    df["latest_order"] =latest_order.value 
    df["recent_visits"] = 1

    DATA.append(df)

def print_info(info, frame, index):

    if(info['recent_visits'] > 2):
        print("단골 손님")

    '''
    print("누적 " , info['recent_visits'] , "회 방문한 손님")
    print('phone number :' , info['phone_number'])
    print('gender :' ,info['gender'])
    print('latest visit date :', info['latest_visit_date'])
    print('latest order', info['latest_order'])
    '''
    total_visits = "total visits : "+str(info['recent_visits'])
    phone_number = "phone number :" + str(info['phone_number'])
    gender = "gender :"+str(info['gender'])
    visit_date = "latest visit date : "+str(info['latest_visit_date'])
    latest_order = "latest order : "+str(info['latest_order'])
    '''
    gap=30
    font = cv2.FONT_HERSHEY_SIMPLEX

    k = 1
    cv2.putText(frame, total_visits, (10, gap*k), font,1,(252,234,92))
    k+=1
    cv2.putText(frame, phone_number, (10, gap*k), font,1,(252,234,92))
    k+=1
    cv2.putText(frame, gender, (10, gap*k), font,1,(252,234,92))
    k+=1
    cv2.putText(frame, visit_date, (10, gap*k), font ,1,(252,234,92))
    k+=1
    cv2.putText(frame, latest_order, (10, gap*k), font,1,(252,234,92))
    '''
    printBox = App(title="information")
    if(info['recent_visits'] > 2):

        message = Text(printBox, "custom") 
    else:
        message = Text(printBox, "customer")
    visitM = Text(printBox, total_visits)
    phoneM = Text(printBox, phone_number)
    genderM = Text(printBox, gender)
    visit_dateM = Text(printBox, visit_date)
    orderM = Text(printBox, latest_order) 
    today_orderM = Text(printBox, "Today's order")
    orderB = TextBox(printBox) 
    printBox.display()
    order_value = orderB.value 
    DATA[index]['latest_order'] =order_value 
    print(order_value)
    print(orderB) 

load_csv()

'''
1. 카메라 실행 
'''

cap = cv2.VideoCapture(-1)

cap.set(cv2.CAP_PROP_FPS,60 )


if cap.isOpened():
	print("open")

'''
2. 5초 마다 반복
    input 폴더 내의 파일 삭제
    이미지 캡처 후 input_path에 저장
    이미지 확인 함수 실행
'''

cnt = 0
while(True):
    ret, frame = cap.read()    # Read 결과와 frame
    cnt += 1

    if(ret) :
        
        if(cnt % 60 == 0):
            check_image(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
        
        '''
        gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)    # 입력 받은 화면 Gray로 변환
        cv2.imwrite('./input/input.jpg',gray)
        check_image()
        cv2.imshow('frame_gray', gray)    # Gray 화면 출력
        if cv2.waitKey(1) == ord('q'):
            break
        '''

cap.release()
cv2.destroyAllWindows()

save_csv()
