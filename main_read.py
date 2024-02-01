import cv2

# Načtení obrázku
img = cv2.imread("1.jpg")

# Konvertování obrázku na GRAYSCALE
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Úpravy GRAYSCALE obrázku na jednotlivé použití pro vyhledávání objektů
_, thresh_defect = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
_, thresh_filtr = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

# Vytvoření kontur pro jednotlivé proměnné
defects, _ = cv2.findContours(thresh_defect, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
filters, _ = cv2.findContours(thresh_filtr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# První cyklus pro zjištění defetku ve filtru s podmínkou upravující vykreslený rozsah px
# aby nedocházelo k zobrazení i filtru bez defektu
for defect in defects:
    area = cv2.contourArea(defect)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(defect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 15)

# Druhý cyklus zobrazující samotný filtr s podmínkami pro vykreslení jen oblasti filtru
for filtr in filters:
    area = cv2.contourArea(filtr)
    if area > 1000:
        x, y, w, h = cv2.boundingRect(filtr)
        if x > 100:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 15)

# Zobrazení obrázku s detekcemi a ukončení
cv2.imshow("detekce", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
