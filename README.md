# RobotComment

Предназначено для использования в Битве Роботов ФБОУ ВО КГЭУ

Системные Требования:
  Требуется Python 3.12.3 и пакеты из requirements.txt или requirementsCUDA.txt 

  Текст установки на ОС WINDOWS
  
Установка:
  1) Установите Python 3.12.3, в установщике рекомендуется выдать права администратора и добавить python.exe в PATH
  2) Скачайте проект и распакуйте в удобную для вас директорию. Убедитесь что в директории установки не используются символы кроме цифр и латиницы
  3) Откройте командную строку от имени администратора и выберите директорию установки (пример для Windows: cd C:\Users\UserName\Folder\RobotComment)
  4) В командную строку введите команду "python -m venv venv" и дождитесь выполнения команды
  5) В командную строку введите команду "venv\Scripts\activate" и убедитесь что перед (слева) директорией появилось (venv) для этого просто нажмите Enter несколько раз
  6) Последний шаг, выберите ОДИН из вариантов ниже (7 или 8)
  7) Если вы хотите использовать видеокарту с поддержкой CUDA, то в командную строку введите команду
       "pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
       После этого введите команду
       "pip install -r requirementsCUDA.txt"
  8) Если вы хотите использовать CPU вместо видеокарты, то в командную строку введите команду "pip install -r requirements.txt"


# --help
usage: main.py [-h] [-cbd [CHECK_BAD_DETECTIONS]] [-dir] [-i INPUT] {track,train,split,label}

positional arguments:
  {track, train, split, label}
                        Track: Track object. Train: Train model. Split: Split video by frames. Label: auto-labeling tool.

options:
  -h, --help            show this help message and exit
  -cbd [CHECK_BAD_DETECTIONS], --check_bad_detections [CHECK_BAD_DETECTIONS]
                        usage: -cbd <expected amount of objects>
  -dir, --directory
  -i INPUT, --input INPUT
                        input device, directory or http/https address
