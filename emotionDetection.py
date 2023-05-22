import cv2
from deepface import DeepFace

# Fotoğrafı yükle
image_path = "C:/ODEV/kizgin.jpg"
image = cv2.imread(image_path)

# Yüz tespiti yap
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Her yüz için duygu analizi yap
for (x, y, w, h) in faces:
    face = image[y:y + h, x:x + w]
    result = DeepFace.analyze(img_path=face, actions=['emotion'])

    # Sonuçları konsola yazdır
    print("Yüz Pozisyonu: ", (x, y, w, h))
    print(result[0])
    print("Duygu Analizi Sonuçları:")
    for emotion, score in result[0]["emotion"].items():
        print(f"{emotion}: {score}")
    
    # Yüzün üzerine duyguyu yazdır
    emotion_label = max(result[0]["emotion"], key=result[0]["emotion"].get)
    cv2.putText(image, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Yüzleri işaretle
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Sonuçları görsel olarak görüntüle
cv2.imshow("Duygu Analizi", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
