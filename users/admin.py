from django.contrib import admin
from .models import Attendance, Time,Present

# Register your models here.
admin.site.register(Time)
admin.site.register(Present)
admin.site.register(Attendance)