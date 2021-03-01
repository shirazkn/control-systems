% Simulation of Epoxy Core Linear Motor with MRAC vs. Static Controller

close all; clear all; 
METHOD = 0; % 'mrac' or 'static'
CONSTRAINTS = 1;
REF_TRAJECTORY = 1;

V_max=3; Umax =5;

FS=18;FS1=16;FS2=14; %scales for text plotting
global Me B Asc kf Acog1 Acog3 omega_y Uu;

% System Parameters 
% Me=0.025;B=0.1;Asc=0.1;kf=1000;Acog1=0.0;Acog3=0.0;omega_y=2*pi/0.06; % Case 1
 Me=0.085;B=0.35;Asc=0.15;kf=1000;Acog1=0.0;Acog3=0.0;omega_y=2*pi/0.06; % Case 2

omega_m=15; zeta_m=1.0; %reference model

% Simulation Parameters
h=0.0005; %sampling period
T=1; dtravel=0.2; % simualtion time span and travel distance
PT=0.6; % half period of the command input

% Controller Parameters
%%
if(METHOD==0)
    k1=100;k2=10; %feedback gains
    gamma_M=0.2; gamma_B=10.0; gamma_F=500.0; %adaptation rate
    
    ym=0.0; ym_dot=0.0;
    theta=[0.055;0.225;0.]; %initial values of parameter estimates theta=[Me_hat,B_hat,Asc_hat]^T
    xc=[ym;ym_dot;theta]; ym_ddot=0.0; %intial controller state
end

%%
if(METHOD==1)
   k1 =-500.0; k2= -2000.0;
   ym_ddot=0.0; sig = 0.0;
end

%%

% Initialization
ym=0; ym_dot=0; uc=dtravel;% initial values of the reference model and command input 
y=0.0; y_dot=0.0; xp=[0;0]; %initial plant outputs and state

S=saturation([-1 1]);%define S as the saturation nonlinearity

for i=1: T/h
   TT(i)=(i-1)*h; %time
   
%calculate the control input Uu(i) based on xc(i), y(i), y_dot(i)
    if(METHOD==0)
        ym=xc(1); ym_dot=xc(2); Me_hat=xc(3); B_hat=xc(4); Asc_hat=xc(5);
        ym_ddot=-omega_m^2*ym-2*zeta_m*omega_m*ym_dot+omega_m^2*uc;
        
        if(REF_TRAJECTORY==1)
           ym = 0.1*(1 - cos(4*pi*TT(i)));
           ym_dot = 0.4*pi*sin(4*pi*TT(i));
           ym_ddot = 1.6*pi^2 *cos(4*pi*TT(i));
        end
        
        em=y-ym;
        em_dot=y_dot-ym_dot;
        s=em_dot+k1*em; 
        Sf=evaluate(S,kf*y_dot); 
        Uu=B_hat*y_dot+Asc_hat*Sf+Me_hat*(ym_ddot-k1*em_dot)-k2*s; %Control input Uu(i) 

        theta=[Me_hat;B_hat;Asc_hat];
        theta0(1,i)=Me; theta0(2,i)=B;theta0(3,i)=Asc; %save true parameter value for plotting

        %Obtain xc_dot(i) and the xc(i+1) for next sampling instance
        xc_dot(1,1)=ym_dot;
        xc_dot(2,1)=ym_ddot;
        xc_dot(3,1)=-gamma_M*(ym_ddot-k1*em_dot)*s;
        xc_dot(4,1)=-gamma_B*y_dot*s;
        xc_dot(5,1)=-gamma_F*Sf*s;
        xc=xc+xc_dot*h; %obtain xc(i+1)
        
        save_theta(:,i)=theta; %save parameter estimates
    end
    
    if(METHOD==1)
        Uu = k1*(y-ym) + k2*sig;
        ym_ddot = -omega_m^2*ym-2*zeta_m*omega_m*ym_dot+omega_m^2*uc;
        ym = ym + ym_dot*h;
        ym_dot = ym_dot + ym_ddot*h;
        sig = sig + (y-ym)*h;
        save_theta(:,i)=[y-ym; sig; 0.0]; %save parameter estimates
        theta0(:,i)=save_theta(:,i);
    end
    
    if(CONSTRAINTS==1)
        if abs(Uu) > Umax
            Uu=Umax*sign(Uu); %Simulate Control Input Saturaion
        end
    end
    
    save_u(i)=Uu; %save control input for plotting
    save_y(i)=y;  %save output
    save_ym(i)=ym; %save the reference output
    save_uc(i)=uc; %save the reference command input

    %Call "ode45" to simulate the plant to obtain y(i+1)
    ti=TT(i); tf=ti+h;
    [t,xy]=ode45('eclMotor1_plant',[ti,tf],xp);
    [NN,MM]=size(xy);
    xp=xy(NN,:); y=xp(1,1);y_dot=xp(1,2);
           %the last output of ode45 is the state at t=(i+1)h
           
    if(CONSTRAINTS==1)
        y=round(xp(1,1)/0.000001)*0.000001; %Simulate the position resolution of 0.000001m
        if abs(xp(1,2))>V_max
            y0_dot=V_max*sign(xp(1,2)); %Simulate maximal measurable speed
        else
            y0_dot=xp(1,2);
        end
        y_dot=round(y0_dot/0.001)*0.001; %Simulate the velocity resolution of 0.001m/sec if (mod(i*h,PT)==0)
    end
    
    if(mod(i*h,PT)==0) 
           NN=floor(i*h/PT); uc=uc+(-1)^NN*dtravel;
    end     %generate the command input uc(i+1)
end

%% Plotting

% Square wave response
subplot(2,1,1), plot(TT,save_uc,':',TT,save_ym,'--',TT,save_y)
xlabel('Time (sec)'); h=get(gca,'xlabel');set(h,'FontSize',FS1); ylabel('Response'); h=get(gca,'ylabel');set(h,'FontSize',FS1); h=gtext('Dotted: uc Dashed: ym Solid: y'); set(h,'FontSize',FS1); set(gca,'FontSize',FS2);
subplot(2,1,2), plot(TT,save_u)
xlabel('Time (sec)'); h=get(gca,'xlabel');set(h,'FontSize',FS1); ylabel('Control Input');h=get(gca,'ylabel');set(h,'FontSize',FS1); set(gca,'FontSize',FS2);
print -depsc MARC_motors_P_yu_C1.eps
pause

% Tracking error
subplot(1,1,1),plot(TT,save_y-save_ym)
xlabel('Time (sec)'); h=get(gca,'xlabel');set(h,'FontSize',FS); ylabel('Tracking Error');h=get(gca,'ylabel');set(h,'FontSize',FS); set(gca,'FontSize',FS);
print -depsc MRAC_motor_P_e_C1.eps
grid; pause

% Estimates of parameters
subplot(2,2,1), plot(TT,theta0(1,:),':',TT,save_theta(1,:));
xlabel('Time (sec)'); h=get(gca,'xlabel');set(h,'FontSize',FS2); ylabel('Estimate of \theta_1'); h=get(gca,'ylabel');set(h,'FontSize',FS2); set(gca,'FontSize',FS2);
subplot(2,2,2), plot(TT,theta0(2,:),':',TT,save_theta(2,:));
xlabel('Time (sec)'); h=get(gca,'xlabel');set(h,'FontSize',FS2); ylabel('Estimate of \theta_2'); h=get(gca,'ylabel');set(h,'FontSize',FS2); set(gca,'FontSize',FS2);
subplot(2,2,3), plot(TT,theta0(3,:),':',TT,save_theta(3,:));
xlabel('Time (sec)'); h=get(gca,'xlabel');set(h,'FontSize',FS2); ylabel('Estimate of \theta_3'); h=get(gca,'ylabel');set(h,'FontSize',FS2); set(gca,'FontSize',FS2);
subplot(2,2,4)
print -depsc MRAC_motor_P_P_C1.eps