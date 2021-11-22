from django.test import LiveServerTestCase
import requests

class UploadTest(LiveServerTestCase):
    def test_upload(self):
        file = open('tests/10-20190207162430645.jpg','rb')
        files = [
            ('imagefile', ('10-20190207162430645.jpg', file, 'image/jpg'))
        ]
        r = requests.post('http://127.0.0.1:8000/client/imageuploads/',
            data={
                'title':'Test Image'
            },
            files=files
        )
        file.close()
        self.assertEqual(201, r.status_code)  # created