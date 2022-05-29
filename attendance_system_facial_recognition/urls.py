"""attendance_system_facial_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from recognition import views as recog_views
from users import views as users_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', recog_views.home, name='home'),
    path('dashboard/', recog_views.dashboard, name='dashboard'),
    path('train/', recog_views.train, name='train'),
    path('login/',auth_views.LoginView.as_view(template_name='users/login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='recognition/home.html'),name='logout'),
    path('register/', users_views.register, name='register'),
    path('mark_your_attendance', recog_views.mark_your_attendance ,name='mark-your-attendance'),
    path('mark_your_attendance_out', recog_views.mark_your_attendance_out ,name='mark-your-attendance-out'),
    path('view_attendance_home', recog_views.view_attendance_home ,name='view-attendance-home'),
    path('view_attendance_date', recog_views.view_attendance_date ,name='view-attendance-date'),
    path('view_attendance_employee', recog_views.view_attendance_employee ,name='view-attendance-employee'),
    path('not_authorised', recog_views.not_authorised, name='not-authorised'),
    path('hand_det',recog_views.index, name='hand-det'),
    path('inattendance',recog_views.index_in,name='in-attendance'),
    path('outattendance',recog_views.index_out,name="out-attendance"),
    path('video',recog_views.video,name='video'),
    path('video2',recog_views.video2,name='video2'),
    path('video3',recog_views.video3,name='video3'),
    path('action_done',recog_views.action,name='action-done'),
    path('video4/<str:username>/',users_views.video4,name='video4'),
    path('marked_in',recog_views.marked_in,name='marked-in'),
    path('marked_out',recog_views.marked_out,name='marked-out'),
    path('tutorial',recog_views.tutorial,name='tutorial'),
    
]
