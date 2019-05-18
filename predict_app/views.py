from django.http import HttpResponse
from django.template import RequestContext,loader
from .models import Results

from django.http import JsonResponse

def index(request):
    return HttpResponse(loader.get_template('mysensor/index.html').render(RequestContext(request,{'latest_results_list':Results.objects.all()})))

def update(request):
     results = [ob.as_json() for ob in Results.objects.all()]
     return JsonResponse({'latest_results_list':results})