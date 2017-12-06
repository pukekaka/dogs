import os

current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    print(current_directory+'/'+'00add40a2459c7075c6db8a25ac6968d.json')
    n = os.path.getsize(current_directory+'/'+'00add40a2459c7075c6db8a25ac6968d.json')
    print(n, "Bytes")
    print(n / 1024, "KB")
    print("%.2f MB" % (n / (1024.0 * 1024.0)))
except os.error:
    print("파일이 없거나 에러입니다.")