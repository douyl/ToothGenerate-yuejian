import numpy as np
import dlib
import cv2


def face_landmark_detect(img, weight_path='./DetectFace/ckpts/shape_predictor_68_face_landmarks.dat'):
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img, 0)
    shape_predictor = dlib.shape_predictor(weight_path)
    try:
        shape = shape_predictor(img, faces[0])
    except:
        return None, None
    landmarks = np.array([(v.x, v.y) for v in shape.parts()])
    return faces[0], landmarks


def DetectFace(img_path, newsize=(512, 512), label_path=None):
    img = cv2.imread(img_path)
    face, landmarks = face_landmark_detect(img)

    y1 = face.top() if face.top() > 0 else 0
    y2 = face.bottom() if face.bottom() > 0 else 0
    x1 = face.left() if face.left() > 0 else 0
    x2 = face.right() if face.right() > 0 else 0

    Face = img[y1:y2+1, x1:x2+1]
    Face = cv2.resize(Face, newsize)

    # Face = cv2.cvtColor(np.asarray(Face), cv2.COLOR_BGR2RGB)


    info = {
        'coord_x': (x1, x2+1),
        'coord_y': (y1, y2+1),
        'face_size': (x2+1-x1, y2+1-y1),
        'new_size': newsize,
    }

    return img, Face, info

if __name__ == '__main__':
    DetectFace(r'C:\IDEA_Lab\Project_tooth_photo\TeethSegm\Data\good/0a31887d774f4d439a87179045255951.jpg')