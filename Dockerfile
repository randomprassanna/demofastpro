FROM python:3.8.5

WORKDIR usr/scr/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "first_file:app", "--host", "0.0.0.0","--port", "8000" ]
