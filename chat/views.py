from django.shortcuts import render
from django.http import HttpResponse

from .models import *
from .forms import *

from .utils import *

# Create your views here.


def index(request):
    form = MessageForm()
    bert, nb, dt = '', '', ''
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            text = form.data['input']
            bert = predict_text_bert(text)
            nb = predict_text_nb(text)
            dt = predict_text_dt(text)

    context = {'form': form, 'bert': bert, 'nb': nb, 'dt': dt}
    return render(request, 'chat/input.html', context)
