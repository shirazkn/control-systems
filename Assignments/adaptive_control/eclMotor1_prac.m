% eclMotor1 with the following limitations:
%   (i)   Position (y) resolution < 10e-6m
%   (ii)  Velocity resolution < 10e-3m/s, Max(abs(vel)) is 3m/s
%   (iii) Control input capped at 5v, sampling rate is 2kHz max

close all; clear all; FS=18;FS1=16;FS2=14; %scales for text plotting
global Me B Asc kf Acog1 Acog3 omega_y Uu; % System Parameters Me=0.025;B=0.1;Asc=0.1;kf=1000;Acog1=0.0;Acog3=0.0;omega_y=2*pi/0.06; %plant omega_m=15; zeta_m=1.0; %reference model
   V_max=3; Umax =5; %maximal measurable speed and control input saturation limit
   %Controller constants
   k1=100;k2=10; %feedback gains
gamma_M=0.2; gamma_B=10.0; gamma_F=500.0; %adaptation rate
  h=0.0005; %sampling period
  T=4; dtravel=0.2; % simualtion time span and travel distance
  PT=0.6; % half period of the command input
% Initialization
ym=0; ym_dot=0; uc=dtravel;% initial values of the reference model and the command input
theta=[0.055;0.225;0.];% initial values of parameter estimates theta=[Me_hat,B_hat,Asc_hat]^T
xc=[ym;ym_dot;theta]; ym_ddot=0.0; %intial controller state
y=0.0; y_dot=0.0; xp=[0;0]; %initial plant outputs and state
S=saturation([-1 1]);%define S as the saturation nonlinearity
  for i=1: T/h
     TT(i)=(i-1)*h; %time
  %calculate the control input Uu(i) based on xc(i), y(i), y_dot(i)
ym=xc(1); ym_dot=xc(2); Me_hat=xc(3); B_hat=xc(4); Asc_hat=xc(5);
em=y-ym;
em_dot=y_dot-ym_dot;
s=em_dot+k1*em;
ym_ddot=-omega_m^2*ym-2*zeta_m*omega_m*ym_dot+omega_m^2*uc; Sf=evaluate(S,kf*y_dot); Uu=B_hat*y_dot+Asc_hat*Sf+Me_hat*(ym_ddot-k1*em_dot)-k2*s; %Control input Uu(i)
if abs(Uu) > Umax
Uu=Umax*sign(Uu); %Simulate Control Input Saturaion
end
save_u(i)=Uu; %save control input for plotting
save_y(i)=y; %save output
save_ym(i)=ym; %save the reference output
save_uc(i)=uc; %save the reference command input theta=[Me_hat;B_hat;Asc_hat];
save_theta(:,i)=theta; %save parameter estimates
theta0(1,i)=Me; theta0(2,i)=B;theta0(3,i)=Asc; %save true parameter value for
  plotting
  %Obtain xc_dot(i) and the xc(i+1) for next sampling instance
      xc_dot(1,1)=ym_dot;
      xc_dot(2,1)=ym_ddot;
      xc_dot(3,1)=-gamma_M*(ym_ddot-k1*em_dot)*s;
      xc_dot(4,1)=-gamma_B*y_dot*s;
      xc_dot(5,1)=-gamma_F*Sf*s;
      xc=xc+xc_dot*h; %obtain xc(i+1)
  %Call "ode45" to simulate the plant to obtain y(i+1)
ti=TT(i); tf=ti+h;
[t,xy]=ode45('eclMotor1_plant',[ti,tf],xp);
[NN,MM]=size(xy);
xp=xy(NN,:); %the last output the ode45 is the state at t=(i+1)h 
y=round(xp(1,1)/0.000001)*0.000001; %Simulate the position resolution of 0.000001m
     if abs(xp(1,2))>V_max
          y0_dot=V_max*sign(xp(1,2)); %Simulate maximal measurable speed
     else
          y0_dot=xp(1,2);
end
y_dot=round(y0_dot/0.001)*0.001; %Simulate the velocity resolution of 0.001m/sec 
if (mod(i*h,PT)==0)
          NN=floor(i*h/PT); uc=uc+(-1)^NN*dtravel;
     end        %generate the command input uc(i+1)
  end
