from django import forms
class QuestionForm(forms.Form):
    test_statement = forms.CharField(label='statement', max_length=200)
    