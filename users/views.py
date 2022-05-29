from django.http import StreamingHttpResponse
from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import os
import dlib
from imutils.face_utils import FaceAligner
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import cv2
from django.contrib.auth.models import User

def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	return False

# REGISTER VIEW
@login_required
def register(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save()
			messages.success(request, f'Employee registered successfully!')
			if username_present(form.cleaned_data.get("username")):
				messages.success(request, f'Dataset Created')
				context = {}
				context['user'] = form.cleaned_data.get("username")
				return render(request,'recognition/create_dataset.html', context)
	else:
		form=UserCreationForm()
		return render(request,'users/register.html', {'form' : form})


# CREATING INITIAL DATASET
@login_required
def add_photos(request,username):
	if username_present(username):
		messages.success(request, f'Dataset Created')
		context = {}
		context['user'] = username
		return render(request,'recognition/add_photos.html', context)
	else:
		messages.warning(request, f'No such username found. Please register employee first.')
		return redirect('dashboard')
			
def video4(request,username):
	return StreamingHttpResponse(create_dataset(request,username),
                    content_type='multipart/x-mixed-replace; boundary=frame')
def create_dataset(request,username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id) #Loading the HOG face detector and shape predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	vs = VideoStream(src=0).start()
	sampleNum = 0
	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #We need a greyscale image for the classifier to work
		faces = detector(gray_frame,0)
		for face in faces:
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			sampleNum = sampleNum+1
			if face is None:
				continue
			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned) #Saving the face in the respected folder
			face_aligned = imutils.resize(face_aligned ,width = 400)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			cv2.waitKey(50)
		ret,buffer=cv2.imencode('.jpg',frame)
		frame=buffer.tobytes()
		yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		cv2.waitKey(1)
		if(sampleNum>100):
			break
	vs.stop()
	cv2.destroyAllWindows()





