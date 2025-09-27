from django.contrib import admin
from .models import AudioRecording

# Register your models here.
@admin.register(AudioRecording)
class Admin(admin.ModelAdmin):
    '''Admin View for '''

    list_display = ('audio_file',)
    # list_filter = ('',)
    # raw_id_fields = ('',)
    # readonly_fields = ('',)
    # search_fields = ('',)
    # date_hierarchy = ''
    ordering = ('created_at',)