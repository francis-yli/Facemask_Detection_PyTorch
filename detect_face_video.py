# detect_face_video
# detect people with facemasks in video
# Author: Yangjia Li (Francis)
# Date: 
# Last Modeified: 

from pathlib import Path
import click
import cv2
import torch
from skvideo.io import FFmpegWriter, vreader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from face_detector import FaceDetector
from train_model import MaskDetector

# add command line calls via click
@click.command(help="""
                    model_path: path to model.ckpt\n
                    video_path: path to video file
                    """)
@click.argument('model_path')
@click.argument('video_path')
@click.option('--output', 'output_path', type=Path,
              help='please specify output path to save the video')
@torch.no_grad() # reduce memory consumption

def tag_facemasks_video(model_path, video_path, output_path=None):
    """ detect if persons in video are wearing masks or not
    """
    model = MaskDetector()
    model.load_state_dict(torch.load(model_path)['state_dict'], strict=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    face_detector = FaceDetector(
        prototype='C:/Users/franc/Documents/Code/Facemask_Detection_PyTorch/deploy.prototxt.txt',
        model='C:/Users/franc/Documents/Code/Facemask_Detection_PyTorch/res10_300x300_ssd_iter_140000.caffemodel',
    )
    
    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),
    ])
    
    if output_path:
        writer = FFmpegWriter(str(output_path))
    
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.namedWindow('facemask detection', cv2.WINDOW_FREERATIO)
    labels = ['No mask', 'Has Mask']
    labelColor = [(10, 0, 255), (10, 255, 0)]
    for frame in vreader(str(video_path)):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect(frame)
        for face in faces:
            xStart, yStart, width, height = face
            
            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)
            
            # predict mask label on extracted face
            face_img = frame[yStart:yStart+height, xStart:xStart+width]
            output = model(transformations(face_img).unsqueeze(0).to(device))
            _, predicted = torch.max(output.data, 1)
            
            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=4)
            
            # center text according to the face frame
            textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2
            
            # draw prediction label
            cv2.putText(frame,
                        labels[predicted],
                        (textX, yStart-20),
                        font, 1, labelColor[predicted], 2)
        if output_path:
            writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('facemask detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if output_path:
        writer.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tag_facemasks_video()