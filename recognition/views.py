from django.shortcuts import render,redirect
from .forms import usernameForm,DateForm,UsernameAndDateForm
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time, Attendance
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
import time
import mediapipe as mp
from tensorflow.keras.models import load_model
from django.http import StreamingHttpResponse
mpl.use('Agg')


def home(request):
	return render(request, 'recognition/index.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		return render(request, 'recognition/admin_page.html')


# UTILITY VIEWS
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	return False

def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)

@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')

def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)

def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])
	except:
		return ([-1],[0])
	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])
	return (result[0],prob[0][result[0]])

def check_validity_times(times_all):
	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)
	if(len(times_in)!=len(times_out)):
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time
	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time
		else:
			prev_time=obj.time
		prev=curr
	return (True,break_hourss)

def convert_hours_to_hours_mins(hours):
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")


# DATABASE ACCESS VIEWS
def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
			qs=Present.objects.get(user=user,date=today)
		except:
			qs= None
	
		if qs is None:
			if present[person]==True:
						a=Present(user=user,date=today,present=True)
						a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=False)
			a.save()

def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()

def present_attendance(flag):
	Attendance.objects.all().delete()
	for i in range(1):
		try:
			qs=Attendance.objects.get()
		except:
			qs=None
		if qs is None:
			a=Attendance(person=flag)
			a.save()
	
def marked_in(request):
	obj = Attendance.objects.latest('id')
	val = obj.person
	if val == 1:
		return redirect('in-attendance')
	else:
		return redirect('home')

def marked_out(request):
	obj = Attendance.objects.latest('id')
	val = obj.person
	if val == 1:
		return redirect('hand-det')
	else:
		return redirect('out-attendance')


def update_action_taken(className):
	obj = Time.objects.latest('id')
	obj.action_taken = className
	obj.save()


# MARKING IN ATTENDANCE
def index_in(request):
	return render(request,"recognition/inatt.html")
def video2(request):
		return StreamingHttpResponse(mark_your_attendance(request),
                    content_type='multipart/x-mixed-replace; boundary=frame')
def mark_your_attendance(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat') 
	svc_save_path="face_recognition_data/svc.sav"	
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False
	flag = 0
	vs = VideoStream(src=0).start()
	start_timee=time.time()
	duration = 6
	while((time.time()-start_timee)<duration):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		for face in faces:
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face) 
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
			if(pred!=[-1]):
				flag = 1
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1
				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					count[pred] = 0
				else:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
			else:
				person_name="unknown"
				flag = 1
		ret,buffer=cv2.imencode('.jpg',frame)
		frame=buffer.tobytes()
		yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	vs.stop()
	cv2.destroyAllWindows()
	update_attendance_in_db_in(present)
	present_attendance(flag)


# MARKING OUT ATTENDANCE
def index_out(request):
	return render(request,"recognition/outatt.html")
def video3(request):
	return StreamingHttpResponse(mark_your_attendance_out(request),
                    content_type='multipart/x-mixed-replace; boundary=frame')
def mark_your_attendance_out(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   
	svc_save_path="face_recognition_data/svc.sav"	
	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')
	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False
	flag = 0
	vs = VideoStream().start()
	start_timee=time.time()
	duration = 6
	while((time.time()-start_timee)<duration):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		for face in faces:
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
			if(pred!=[-1]):
				flag = 1
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1
				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					count[pred] = 0
				else:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
			else:
				person_name="unknown"
		ret,buffer=cv2.imencode('.jpg',frame)
		frame=buffer.tobytes()
		yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
	vs.stop()
	cv2.destroyAllWindows()
	update_attendance_in_db_out(present)
	present_attendance(flag)


# DETECTING HAND-GESTURE FOR ACTION
def tutorial(request):
	return render(request,"recognition/tutorial.html")
def index(request):
	return render(request,"recognition/hand_det.html")
def video(request):
	return StreamingHttpResponse(handdet(request),
                    content_type='multipart/x-mixed-replace; boundary=frame')
def handdet(request):
	mpHands = mp.solutions.hands
	hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
	mpDraw = mp.solutions.drawing_utils
	model = load_model('recognition/mp_hand_gesture') 
	f = open('recognition/gesture.names', 'r') 
	classNames = f.read().split('\n')
	f.close()
	capture_duration = 8
	flag = 1
	cap = VideoStream(src=0).start()
	start_time = time.time() 
	while((time.time()-start_time)<capture_duration):
		success = True
		frame=cap.read()
		if not success:
			break
		else:
			global className
			frame = cap.read()
			x, y, c = frame.shape
			framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			result = hands.process(framergb)
			className = '' 
			if result.multi_hand_landmarks:
				landmarks = []
				for handslms in result.multi_hand_landmarks:
					for lm in handslms.landmark:
						lmx = int(lm.x * x)
						lmy = int(lm.y * y)
						landmarks.append([lmx, lmy])
				mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
				prediction = model.predict([landmarks])
				classID = np.argmax(prediction)
				className = classNames[classID]
				if flag == 1:
					flag = 2
			cv2.putText(frame, className, (60, 100), cv2.FONT_HERSHEY_DUPLEX, 
                            1, (0,0,255), 2, cv2.LINE_AA)
			ret,buffer=cv2.imencode('.jpg',frame)
			frame=buffer.tobytes()
			yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
	cap.stop()
	cv2.destroyAllWindows()
	update_action_taken(className)
def action(request):
	return render(request,"recognition/action_done.html")


# TRAINING MODEL FOR NEW RECORDS
@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	training_dir='face_recognition_data/training_dataset'
	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count+=1
	X=[]
	y=[]
	i=0
	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				y.append(person_name)
				i+=1
			except:
				os.remove(imagefile)
	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)
	messages.success(request, f'Training Complete.')
	return render(request,"recognition/home.html")


# ADMIN SIDE EMPLOYEE DATA
@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})

@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None
	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				u=User.objects.get(username=username)
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')

			else:
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')
	else:
			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})

@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)
				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')
	else:
			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})

def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs
	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		obj1 = time_qs.filter(user_id=obj.user_id)
		for x in obj1:
			str = x.action_taken
		obj.action_taken=str
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss
		else:
			obj.break_hours=0
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
	df = read_frame(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df['break_hours']=df_break_hours
	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs

def hours_vs_date_given_employee(present_qs,time_qs,admin=True):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs
	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss
		else:
			obj.break_hours=0
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
	df = read_frame(qs)	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.close()
	return qs


# UNUSED VIEWS
def this_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))
	while(cnt<5):
		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)
	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
	sns.lineplot(data=df,x='date',y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()

def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))
	while(cnt<5):
		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)
			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)
	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["emp_count"]=emp_cnt_all
	sns.lineplot(data=df,x='date',y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()