#include<Servo.h>
#include<Mouse.h>

                                                    //Functions

                          //Arm motion

//Arm function declaration
void arm_motion(Servo servoZ_Right, Servo servoXY_Right, Servo servoZ_Left, Servo servoXY_Left, int Delay, int angleZ, int angleXY);

//Arm function definition
void arm_motion(Servo servoZ_Right, Servo servoXY_Right, Servo servoZ_Left, Servo servoXY_Left, int Delay, int angleZ, int angleXY){
  servoZ_Right.write(angleZ);
  servoZ_Left.write(180 - angleZ);
  servoXY_Right.write(0);
  servoXY_Left.write(180);

  delay(Delay);

  for(int i = angleZ; i >= 0; i--){
    servoZ_Right.write(i);
    servoZ_Left.write(180 - i);
    delay(15);
  }
  //servoZ_Right.write(0);
  //servoZ_Left.write(180);

  delay(Delay);

  for(int i = 0; i <= angleXY; i++){
    servoXY_Right.write(i);
    servoXY_Left.write(180 - i);
    delay(10);
  }
  //servoXY_Right.write(angleXY);
  //servoXY_Left.write(180 - angleXY);

  delay(Delay);

  for(int i = 0; i <= angleZ/2; i++){
    servoZ_Right.write(i);
    //servoXY_Right.write(angleXY + i/25);
    servoZ_Left.write(180 - i);
    //servoXY_Left.write(180 - angleXY - i/25);
    delay(10);
  }

  delay(Delay/2);
  
  for(int i = angleZ/2; i <= angleZ; i++){
    servoZ_Right.write(i);
    servoXY_Right.write(angleXY + angleZ/(2 * 25) + i/5);
    servoZ_Left.write(180 - i);
    servoXY_Left.write(180 - angleXY - angleZ/(2 * 25) - i/5);
  }

  delay(Delay);

  for(int i = angleXY + (angleZ/(2*20) + angleZ/(2*5)); i >= 0; i--){
    servoXY_Right.write(i);
    servoXY_Left.write(180 - i);
    delay(30);
  }
  //servoXY_Right.write(0);
  //servoXY_Left.write(180);

  delay(Delay);
}
                            //Arm motion end


                            //Sonar distance 

//Sonar distance measuring function declaration
int sonarDistance(int echoPin, int trigPin);

//Sonar distance measuring function definition
int sonarDistance(int echoPin, int trigPin){
  //Generating sonar pulse
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  //Measuring Echo Pulse duration
  long duration = pulseIn(echoPin, HIGH);

  //Calculating distance in centimeters
  int distance = duration / 58.2;

  return distance;
}
                          //Sonar distance end

                                                          //Functions end



                                                          //Global variables declaration
//Servo variables and pins
Servo servoZ_Right, servoXY_Right;     //Right arm
Servo servoZ_Left, servoXY_Left;       //Left arm

int angleZ = 100;                      //Z axis angle
int angleXY = 35;                      //XY axis angle
int DelayArm = 1500;                   //Delay time

const int servoZ_Left_Pin = 11;
const int servoXY_Left_Pin = 10;
const int servoZ_Right_Pin = 9;
const int servoXY_Right_Pin = 8;

//Sonar pins
const int sonarTrigPin = 7;            //Sonar Trigger Pin
const int sonarEchoPin = 6;            //Sonar Echo Pin

                                                          //Global variables declaration end

                                                          //Working variables
int count = 0;
                                                          //Working variables end


                                                          //Main code

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  //Servo arms

  //Right arm setup
  servoZ_Right.attach(servoZ_Right_Pin);
  servoXY_Right.attach(servoXY_Right_Pin);

  servoZ_Right.write(angleZ);
  servoXY_Right.write(0);

  //Left arm setup
  servoZ_Left.attach(servoZ_Left_Pin);
  servoXY_Left.attach(servoXY_Left_Pin);

  servoZ_Left.write(180 - angleZ);
  servoXY_Left.write(180);


  //Sonar setup
  pinMode(sonarTrigPin, OUTPUT);    //Setting Sonar Output Pin
  pinMode(sonarEchoPin, INPUT);     //Setting Sonar Input Pin

}


void loop() {
  // put your main code here, to run repeatedly:
  /*int distance = sonarDistance(sonarEchoPin, sonarTrigPin);

  if(distance <= 3){
    arm_motion(servoZ_Right, servoXY_Right, servoZ_Left, servoXY_Left, DelayArm, angleZ, angleXY);
  }*/

  for( ; count < 1; count++){
    arm_motion(servoZ_Right, servoXY_Right, servoZ_Left, servoXY_Left, DelayArm, angleZ, angleXY);
  }
}
