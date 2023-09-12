#include<Servo.h>

Servo servo_test;

void setup() {
  // put your setup code here, to run once:
  servo_test.attach(8);
  servo_test.write(0);
}

void loop() {
  // put your main code here, to run repeatedly:
  for(int i = 0; i <= 90; i++){
    servo_test.write(i);
    delay(15);
  }

  delay(2000);

    for(int i = 90; i <= 180; i++){
    servo_test.write(i);
    delay(15);
  }

  delay(2000);

  for(int i = 180; i >= 90; i--){
    servo_test.write(i);
    delay(15);
  }

  delay(2000);

    for(int i = 90; i >= 0; i--){
    servo_test.write(i);
    delay(15);
  }

  delay(2000);
}
