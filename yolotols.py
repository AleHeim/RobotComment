
import zipfile


# Создайте клиента Label Studio
LABEL_STUDIO_URL = 'http://robot-fight.ru:9000/'
API_KEY = '9cf7a40da850b23f8f63927087a089f05fd61d37'

from label_studio_sdk import Client

ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()
project = ls.get_project(11)
project.export_tasks(export_type='YOLO', download_all_tasks = True, download_resources = True, export_location='./datasets/LS EXPORT/dataset.zip')
with zipfile.ZipFile('./datasets/LS EXPORT/dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./datasets/LS EXPORT/dataset')
    zip_ref.close()

# Получите проект и набор данных
# project = client.get_project(project_id="<project_id>")
# dataset = project.get_dataset(dataset_id="<dataset_id>")

# Загрузите файлы без переименования
# dataset.import_from_dir(
   # input_dir="/path/to/data/directory",
   # preserve_filenames=True,
# )

# label-studio-converter import yolo -i ./datasets/one -o ./datasets/one/ls-tasks.json --image-root-url "/data/upload/11/"
   
