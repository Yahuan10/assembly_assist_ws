#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

// Gerätename für die RECHTE Hand
const char* deviceName = "Motor_Right";

const int motorPin = 7; // GPIO-Pin für den Motor (bitte anpassen, falls nötig)
const unsigned int vibrationDuration = 4000; // 4 Sekunden Vibration

// Globale Variablen für BLE und Vibrationssteuerung
BLEServer* pServer = nullptr;
BLECharacteristic* pDistanceCharacteristic = nullptr;
unsigned long vibrationStartTime = 0;
bool isVibrating = false;

// Funktion, um eine einzigartige UUID aus dem Gerätenamen zu generieren
BLEUUID generateUUID(const char* base, int modifier) {
  uint32_t hash = 0;
  while (*base) {
    hash = (hash * 31) + *base++;
  }
  hash += modifier;
  
  char uuidStr[37];
  snprintf(uuidStr, sizeof(uuidStr), "0000%04x-9a3d-40cf-a152-60b9b0201e3d", hash & 0xFFFF);
  return BLEUUID(uuidStr);
}

// Startet oder aktualisiert die Vibration basierend auf der Distanz
void handleVibration(float distance) {
  if (distance < 0.0f || distance > 1.0f) return; // Wertebereich prüfen

  int pwmValue = 0;
  if (distance <= 0.10f) {
    pwmValue = 255; // Kritisch -> Starke Vibration
  } else if (distance <= 0.20f) {
    pwmValue = 192; // Nah -> Mittlere Vibration
  } else if (distance <= 0.40f) {
    pwmValue = 128; // Mittel -> Leichte Vibration
  } else {
    pwmValue = 64;  // Fern -> Sehr leichte Vibration
  }
  
  analogWrite(motorPin, pwmValue);
  vibrationStartTime = millis();
  isVibrating = true;
}

// Callback-Klasse für eingehende Daten
class MyCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pCharacteristic) {
    // KORREKTUR HIER: 'std::string' zu 'String' geändert
    String value = pCharacteristic->getValue();
    
    if (value.length() == 4) { // Ein Float-Wert besteht aus 4 Bytes
      float receivedDistance;
      memcpy(&receivedDistance, value.c_str(), sizeof(float));
      
      Serial.print("Distanz für RECHTE Hand empfangen: ");
      Serial.println(receivedDistance);
      
      handleVibration(receivedDistance);
    }
  }
};

// Callback-Klasse für Verbindungsstatus
class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    Serial.println("Client verbunden.");
  }

  void onDisconnect(BLEServer* pServer) {
    Serial.println("Client getrennt. Starte Werbung neu...");
    BLEDevice::startAdvertising(); // Werbung neu starten, damit eine neue Verbindung möglich ist
  }
};

void setup() {
  Serial.begin(115200);
  pinMode(motorPin, OUTPUT);
  analogWrite(motorPin, 0); // Motor am Anfang aus

  // UUIDs aus dem Gerätenamen generieren
  BLEUUID serviceUUID = generateUUID(deviceName, 0);
  BLEUUID characteristicUUID = generateUUID(deviceName, 1);

  // BLE initialisieren
  BLEDevice::init(deviceName);
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());
  BLEService* pService = pServer->createService(serviceUUID);

  // Charakteristik für den Distanzwert erstellen
  pDistanceCharacteristic = pService->createCharacteristic(
                              characteristicUUID,
                              BLECharacteristic::PROPERTY_WRITE
                            );
  pDistanceCharacteristic->setCallbacks(new MyCallbacks());

  // Service starten und Werbung beginnen
  pService->start();
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(serviceUUID);
  BLEDevice::startAdvertising();
  
  Serial.print(deviceName);
  Serial.println(" BLE-Server gestartet und wartet auf Verbindung.");
}

void loop() {
  // Stoppt die Vibration nach der festgelegten Dauer
  if (isVibrating && (millis() - vibrationStartTime >= vibrationDuration)) {
    analogWrite(motorPin, 0); // Motor stoppen
    isVibrating = false;
  }
}