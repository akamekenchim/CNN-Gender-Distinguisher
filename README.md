Mở Google Colab, dán đoạn code python vào, sau đó truy cập link drive 
https://drive.google.com/file/d/1m72KXXvr7jaHHO5xAKrfruOEJU8mATp1/view?usp=sharing
để copy file .pth về drive của mình. Sau đó thay đường dẫn trong torch.load bằng
đường dẫn đến file .pth trong drive (cần mount GG Drive to runtime trước).
Thay đường dẫn trong ImageFolder(root="..") bằng folder ảnh của bạn.
