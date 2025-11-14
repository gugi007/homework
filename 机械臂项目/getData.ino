#include <Wire.h>
#include <ESP8266WiFi.h>
#include <ESP8266WiFiMulti.h>
#include "I2Cdev.h"
#include "MPU6050.h"
MPU6050 accelgyro;//实例化
//首先启用定时器库和信号量库
#define PT_USE_TIMER
#define PT_USE_SEM
const char* ssid = "AP热点";
const char* password = "123456789";
const char* server_ip = "192.168.4.1"; // 修改为实际的服务器IP地址
unsigned long now, lastTime = 0;
float dt;                                   //微分时间
int16_t ax, ay, az, gx, gy, gz;             //加速度计陀螺仪原始数据
float aax=0, aay=0,aaz=0, agx=0, agy=0, agz=0;    //角度变量
long axo = 0, ayo = 0, azo = 0;             //加速度计偏移量
long gxo = 0, gyo = 0, gzo = 0;             //陀螺仪偏移量

float pi = 3.1415926;
float AcceRatio = 16384.0;                  //加速度计比例系数
float GyroRatio = 131.0;                    //陀螺仪比例系数
 
uint8_t n_sample = 8;                       //加速度计滤波算法采样个数
float aaxs[8] = {0}, aays[8] = {0}, aazs[8] = {0};         //x,y轴采样队列
long aax_sum, aay_sum,aaz_sum;                      //x,y轴采样和
 
float a_x[10]={0}, a_y[10]={0},a_z[10]={0} ,g_x[10]={0} ,g_y[10]={0},g_z[10]={0}; //加速度计协方差计算队列
float Px=1, Rx, Kx, Sx, Vx, Qx;             //x轴卡尔曼变量
float Py=1, Ry, Ky, Sy, Vy, Qy;             //y轴卡尔曼变量


//建立数据结构
typedef struct XYZ{
  float pitch;//储存俯仰角，绕x轴旋转
  float roll;//储存翻滚角，绕y轴旋转
  
} XYZ;

XYZ data;//储存相关数据


//uint8_t broadcastAddress[] ={0xEA, 0xDB, 0x84, 0x96, 0x72, 0x17};//ap模式mac地址
int server_port = 80;
WiFiServer server(server_port);//设置端口
WiFiClient client; // 新增一个全局的客户端变量
bool sendData = false;

void setup() {
  Serial.begin(9600);
    {
      Serial.println();
      Serial.print("please dont move ur mpu6050 in 5 secs");
    }
  
  Wire.begin();                      // Initialize comunication
  Wire.beginTransmission(0x68);       // Start communication with MPU6050 // MPU=0x68
  Wire.write(0x6B);                  // Talk to the register 6B
  Wire.write(0x00);                  // Make reset - place a 0 into the 6B register
  Wire.endTransmission(true);        //end the transmission
  
   
    accelgyro.initialize();                 //初始化
 
    unsigned short times = 200;             //采样次数
    for(int i=0;i<times;i++)
    {
        accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz); //读取六轴原始数值
        axo += ax; ayo += ay; azo += az;      //采样和
        gxo += gx; gyo += gy; gzo += gz;
    
    }
    
    axo /= times; ayo /= times; azo /= times; //计算加速度计偏移
    gxo /= times; gyo /= times; gzo /= times; //计算陀螺仪偏移

WiFi.mode(WIFI_STA);
WiFi.begin(ssid, password);
//WiFi.begin(ssid, password);
//设置设备的ip地址
 while (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection failed!");
    delay(1000);
  }

  Serial.print("Connected, IP Address: ");
  Serial.println(WiFi.localIP());

  int server_port = 80;
  client.connect(server_ip, server_port);
  Serial.println("Connected to station");
}

void loop() {
if (accelgyro.testConnection()) {
    unsigned long now = millis();             //当前时间(ms)
    dt = (now - lastTime) / 1000.0;           //微分时间(s)
    lastTime = now;                           //上一次采样时间(ms)
 
    accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz); //读取六轴原始数值

Serial.print("ax::");
Serial.print(ax,1);
Serial.print(" /ay: ");
Serial.print(ay,1);
Serial.print("/az:");
Serial.println(az,1);
//Serial.println(dataStr);

Serial.print("gx::");
Serial.print(gx,1);
Serial.print(" /gy: ");
Serial.print(gy,1);
Serial.print("/gz:");
Serial.println(gz,1);
//Serial.println(dataStr);
 
    float accx = ax / AcceRatio;              //x轴加速度
    float accy = ay / AcceRatio;              //y轴加速度
    float accz = az / AcceRatio;              //z轴加速度
    data.az=accz;
    
    aax = atan(accy / accz) * (-180) / pi;    //y轴对于z轴的夹角
    aay = atan(accx / accz) * 180 / pi;       //x轴对于z轴的夹角
    aaz = atan(accz / accy) * 180 / pi;       //z轴对于y轴的夹角
 
    aax_sum = 0;                              // 对于加速度计原始数据的滑动加权滤波算法
    aay_sum = 0;
    aaz_sum = 0;
  
    for(int i=1;i<n_sample;i++)
    {
        aaxs[i-1] = aaxs[i];
        aax_sum += aaxs[i] * i;
        aays[i-1] = aays[i];
        aay_sum += aays[i] * i;

    
    }
    
    aaxs[n_sample-1] = aax;
    aax_sum += aax * n_sample;
    aax = (aax_sum / (11*n_sample/2.0)) * 9 / 7.0; //角度调幅至0-90°
    aays[n_sample-1] = aay;                        //此处应用实验法取得合适的系数
    aay_sum += aay * n_sample;                     //本例系数为9/7
    aay = (aay_sum / (11*n_sample/2.0)) * 9 / 7.0;

 
    float gyrox = - (gx-gxo) / GyroRatio * dt; //x轴角速度
    float gyroy = - (gy-gyo) / GyroRatio * dt; //y轴角速度
    float gyroz = - (gz-gzo) / GyroRatio * dt; //y轴角速度
    agx += gyrox;                             //x轴角速度积分
    agy += gyroy;                             //x轴角速度积分

    
    /* kalman start */
    Sx = 0; Rx = 0;
    Sy = 0; Ry = 0;
    Sz = 0; Rz = 0;
    
    for(int i=1;i<10;i++)
    {                 //测量值平均值运算
        a_x[i-1] = a_x[i];                      //即加速度平均值
        Sx += a_x[i];
        a_y[i-1] = a_y[i];
        Sy += a_y[i];

    
    }
    
    a_x[9] = aax;
    Sx += aax;
    Sx /= 10;                                 //x轴加速度平均值
    a_y[9] = aay;
    Sy += aay;
    Sy /= 10;                                 //y轴加速度平均值
    a_z[9] = aaz;
    Sz += aaz;
    Sz /= 10;
 
    for(int i=0;i<10;i++)
    {
        Rx += sq(a_x[i] - Sx);
        Ry += sq(a_y[i] - Sy);
        Rz += sq(a_z[i] - Sz);
    
    }
    
    Rx = Rx / 9;                              //得到方差
    Ry = Ry / 9;                        
    Rz = Rz / 9;
  
    Px = Px + 0.0025;                         // 0.0025在下面有说明...
    Kx = Px / (Px + Rx);                      //计算卡尔曼增益
    agx = agx + Kx * (aax - agx);             //陀螺仪角度与加速度计速度叠加
    Px = (1 - Kx) * Px;                       //更新p值
 
    Py = Py + 0.0025;
    Ky = Py / (Py + Ry);
    agy = agy + Ky * (aay - agy); 
    Py = (1 - Ky) * Py;
  
 

  data.pitch=agx;
  data.roll=agy;

// 将XYZ结构体转换为字符串
String dataStr = String(data.pitch) + "," + String(data.roll);
Serial.print("pitch::");
Serial.print(data.pitch,1);
Serial.print(" /roll: ");
Serial.print(data.roll,1);

//Serial.println(dataStr);

  // 发送数据到服务器
  if (client.connected() && !sendData) {
    sendDataAsync(dataStr); // 异步发送数据
    sendData = true;
  }

  // 检查是否已经发送完数据
  if (sendData && !client.connected()) {
    Serial.println("Data sent");
    sendData = false;
    delay(1000); // 延时1秒钟
  }


delay(100); // 延时1秒钟
}
else{
  accelgyro.initialize();
  Serial.print("MPU6050连接失败,错误代码：");
  Serial.println("error");
  delay(3000);
}


}
void sendDataAsync(const String& preData) {
  // 将数据转换为字节数组
  const char* payload = preData.c_str();
  size_t length = strlen(payload);

  // 发送数据
  if (client.connected()) {
    client.write(reinterpret_cast<const uint8_t*>(payload), length);
  }
}

