

# Создайте клиента Label Studio
LABEL_STUDIO_URL = 'http://robot-fight.ru:9000/'
API_KEY = '9cf7a40da850b23f8f63927087a089f05fd61d37'

from label_studio_sdk import Client

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()
project = ls.get_project(10)
project.import_tasks([
    {'image': f'https://developers.google.com/static/drive/images/drive-intro.png'}
])
# Получите проект и набор данных
# project = client.get_project(project_id="<project_id>")
# dataset = project.get_dataset(dataset_id="<dataset_id>")

# Загрузите файлы без переименования
# dataset.import_from_dir(
   # input_dir="/path/to/data/directory",
   # preserve_filenames=True,
# )
   
