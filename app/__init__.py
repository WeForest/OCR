from flask import Flask, request, jsonify
#from werkzeug import secure_filename
#from PIL import Image

app = Flask(__name__)
app.debug = True

# 이미지 업로드
@app.route('/fileUpload', methods=['POST'])
def upload_file():
    data = request.fiels['image']   # data로 된 데이터 가져옴
    #data.save(secure_filename(data.filename))
    
    d = {'success' : True, 'conference' : 'conferenceName', 'name':'idonknow'}
    return jsonify(d)

# 혐오, 욕설 검출
@app.route('/abuse', methods=['GET'])
def abuse():
    data = request.args['data']   # data로 된 데이터 가져옴
    
    #if data == 0 :
    #    return None
    return str(False)



app.run(host='127.0.0.1', port=5000)