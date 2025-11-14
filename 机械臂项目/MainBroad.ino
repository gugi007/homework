#include <Servo.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include<math.h>

Servo right, left, bottom;

#define RightMachine D5
#define LeftMachine D2
#define BottomMachine D6


const char* ssid = "AP热点";      // 替换为您自己的热点名称
const char* password = "123456789";  // 替换为您自己的热点密码
const int serverPort = 80;//端口号
IPAddress serverIP(192,168,100,102); // 替换为您想要设置的IP地址
IPAddress gateway(192, 168, 1, 1); // 替换为您的网关IP地址
IPAddress subnet(255, 255, 255, 0); // 替换为您的子网掩码


WiFiServer server(serverPort);
//WiFiClient client;
WiFiClient client = server.available();

typedef struct {
  float pitch;//绕x轴
  float roll;//绕y轴


} XYZ;

XYZ data; // 

void setup() {
   Serial.begin(9600);
    setServo();
   WiFi.mode(WIFI_AP);//将arduino d1设置为ap模式
  // 设置网络
  //WiFi.softAPConfig(serverIP, gateway, subnet);
  WiFi.softAP(ssid, password);
  

  Serial.print("Access Point: ");    // 通过串口监视器输出信息
  Serial.println(ssid);              // 告知用户NodeMCU所建立的WiFi名
  Serial.print("IP address: ");      // 以及NodeMCU的IP地址
  Serial.println(WiFi.softAPIP());   // 通过调用WiFi.softAPIP()可以得到NodeMCU的IP地址

   
  // 启动服务器
  //server.begin();
  server.begin(serverPort);//固定的端口号启动
  Serial.print("AP IP address: ");
  Serial.println(WiFi.softAPIP());  // 打印AP的IP地址
  
  Serial.println("Server started");
}


void loop() {
  client = server.available();
  
  if (client) {
    Serial.println("New client connected");
    // 设置超时时间为1秒，避免无限等待
    client.setTimeout(1000);

    // 处理与客户端的通信
    while (client.connected()) {
      // 读取客户端发送的数据
      if (client.available()) {
        String request = client.readStringUntil('\r');
        Serial.println(request);
        SplitStr(request, data.pitch, data.roll,data.yaw,data.az);
        Serial.print("Pitch: " + String(data.pitch));
        Serial.print("/Roll: " + String(data.roll));
        Serial.print("/yaw: " + String(data.yaw));
        Serial.println("/yaw: " + String(data.az));
        servomoveRight();
        servomoveBottom();
        delay(100);
      }
     
      // 检查客户端是否断开连接
      if (!client.connected()) {
        Serial.println("Client disconnected");
        break;
      }
    }

    // 关闭与客户端的连接
    client.stop();
  }
  else {
    Serial.println("Server not started");
  }

  delay(1000);
}



void servomoveBottom() {
 float tyaw =data.yaw;
  int current=right.read();
  if((abs(tyaw)<20||current<=-40)||current>=90)return;//如果摆动过小直接返回
  if(tyaw<0&&current>-40)right.write(--current);
  else if(tyaw>0&&current<=90)right.write(++current);
  delay(50);
    
}

void servomoveRight()
{
  float troll =data.roll;
  int current=right.read();
  if((abs(troll)<15||current<=15)||current>=120)return;//如果摆动过小直接返回
  if(troll<0&&current>20)right.write(--current);
  else if(troll>0&&current<=120)right.write(++current);
  delay(50);
 
}

void setServo() {
  bottom.attach(BottomMachine, 500, 2500);
  left.attach(LeftMachine, 500, 2500);
  right.attach(RightMachine, 500, 2500);

  bottom.write(8);
  right.write(70);
  left.write(90);

  Serial.println(left.read());
}

void SplitStr(String receivedData, float& X, float& Y)
{

// 解析数据
XYZ data;
int commaIndex1 = receivedData.indexOf(',');


X = receivedData.substring(0, commaIndex1).toFloat();
Y = receivedData.substring(commaIndex1 + 1, commaIndex2).toFloat();

}
