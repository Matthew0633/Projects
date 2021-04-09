# <br/> 영유아 위험상황 탐지 및 경고 모델('Infant risk detection and warning model')
## '고객 데이터 분석 시스템 구축을 위한 빅데이터 전문가 양성과정' 3차 PROJECT

## 1. '고객 데이터 분석 시스템 구축을 위한 빅데이터 전문가 양성과정'
  
서울산업진흥원(SBA)과 한국능률협회가 공동으로 진행한 교육인 '고객 데이터 분석 시스템 구축을 위한 빅데이터 전문가 양성과정'는 8월 말부터 11월 중순까지 진행되었으며, 파이썬과 머신러닝, 딥러닝 뿐만 아니라 R과 리눅스, SQL 등에 대해 다양한 교육을 진행하였습니다.

*관련기사
- [SBA-능률협회, '빅데이터 전문가 양성과정' 절찬 진행중](https://m.etnews.com/20191115000103?obj=Tzo4OiJzdGRDbGFzcyI6Mjp7czo3OiJyZWZlcmVyIjtOO3M6NzoiZm9yd2FyZCI7czoxMzoid2ViIHRvIG1vYmlsZSI7fQ%3D%3D)
- [SBA-능률협회, 빅데이터 신사업 기획자 양성](https://www.dailygrid.net/news/articleView.html?idxno=306146)

## 2. 영유아 위험상황 탐지 및 경고 모델('Detecting Danger of Baby and Warning Model')  
**설명 하단에 발표자료 포함**  
  
팀원 : 박민형, 김찬용, 정희철  
  
**[해당 프로젝트 수행 동기]**  
저희 팀은 세번째 프로젝트로 '영유아 위험상황 탐지 및 경고 모델'을 진행하였습니다. (2019. 11. 4. ~ 2019. 11. 7.)

걸음마기 아이들의 주택 사고가 가장 잦고, 부모가 육아와 가사를 병행할 때 아이의 행동으로 인한 찰나의 위험 상황이 필수적으로 일어나므로 이를 방지하고 아이의 안전을 지키고자 프로젝트를 수행하게 되었습니다.  

[2018년 어린이 안전사고 동향분석 결과 보고](https://www.kca.go.kr/home/board/download.do?menukey=4062&fno=10024005&bid=00000146&did=1002809259)

**[모델의 목적]**  
저희 모델은 총 4가지의 위험상황 탐지시('위험', '안전') 엄마의 녹음된 음성(아이의 잇따른 행동을 방지하는 메세지)을 출력하고 아이의 후속 위험행동을 방지합니다.

- 이물질 삼킴 : 아이가 물건을 손으로 집어 입 주변으로 가져갈 때 **이물질 삼킴위험** (위험상황)
- 집안 각 방 문앞 : 아이가 각 방의 문 앞에 있을 시 **문끼임사고 위험** (위험상황)
- 부엌 진입 : 아이가 부엌에 진입할 시 위험물건으로 인한 **사고 또는 화상 위험**(위험상황)
- 추락 위험 : 아이가 소파 등에 올라갈 시 **추락 위험** (위험상황)

**[데이터와 모델 학습]**  
사용 데이터는 영상을 찍어 프레임 추출을 이용하였습니다. 특징은 아래와 같습니다.  
  
- 이물질삼킴(19.8G - 다양한 물건과 각도, 위치 이미지)
- 집안 각 방 문앞(2.7G - 다양한 각도, 위치를 가진 아이의 이미지)
- 부엌 진입(3.7G - 다양한 각도, 위치를 가진 아이의 이미지)
- 추락 위험(1.68G - 다양한 각도, 위치를 가진 아이의 이미지)
  
  
대용량 이미지 학습이 요구되어 GCP(Google Cloud Platform)를 이용하여 VM에서 jupyter lab을 통해 학습을 진행하였습니다.  
[Google Cloud](https://cloud.google.com/)

  
**[중요 파일]**  
- `01. Modeling and Fitting CNN models(including others)` : CNN과 keras app의 다양한 CNN모델들을 이용하여 Modeling을 진행하였습니다.
  
- `02. Detecting Danger of Baby and Warning Model` : '집안에 카메라가 아이를 상시 촬영하고 있다는 가정하에' 학습되지 않은 특정 테스트 이미지를 주어 위험상황 탐지와 엄마의 음성 출력 기능 수행을 구현한 코드입니다.
  
## 3. Project Presentation

<img src = '/slides/slide1.PNG'>
<img src = '/slides/slide2.PNG'>
<img src = '/slides/slide3.PNG'>
<img src = '/slides/slide4.PNG'>
<img src = '/slides/slide5.PNG'>
<img src = '/slides/slide6.PNG'>
<img src = '/slides/slide7.PNG'>
<img src = '/slides/slide8.PNG'>
<img src = '/slides/slide9.PNG'>
<img src = '/slides/slide10.PNG'>
<img src = '/slides/slide11.PNG'>
<img src = '/slides/slide12.PNG'>
<img src = '/slides/slide13.PNG'>
<img src = '/slides/slide14.PNG'>
<img src = '/slides/slide15.PNG'>
<img src = '/slides/slide16.PNG'>
<img src = '/slides/slide17.PNG'>
<img src = '/slides/slide18.PNG'>
<img src = '/slides/slide19.PNG'>
<img src = '/slides/slide20.PNG'>
<img src = '/slides/slide21.PNG'>
<img src = '/slides/slide22.PNG'>
<img src = '/slides/slide23.PNG'>
<img src = '/slides/slide24.PNG'>
<img src = '/slides/slide25.PNG'>
<img src = '/slides/slide26.PNG'>
<img src = '/slides/slide27.PNG'>
<img src = '/slides/slide28.PNG'>
