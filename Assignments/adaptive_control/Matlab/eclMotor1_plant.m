% Simulates the plant, used with ode45 to obtain plant state at next
% sampling instance
function xdot=plant(t,xy)
global Me B Asc kf Acog1 Acog3 omega_y Uu; y=xy(1); y_dot=xy(2);
xdot(1,1)=y_dot;
S=saturation([-1 1]); xdot(2,1)=(Uu-B*y_dot-Asc*evaluate(S,kf*y_dot)-Acog1*sin(omega_y*y)- Acog3*sin(3*omega_y*y))/Me;