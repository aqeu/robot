#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#define gps_buffer_size 256
#define sensor_count 4
#define state_count 5
#define motor_max_pwm 255
#define kalman_states 4
#define tcp_buffer_size 512
#define sample_rate 100

typedef struct {
    double lat;
    double lon;
    double alt;
    double spd;
    double hdg;
    long ts;
} gps_data;

typedef struct {
    double x;
    double y;
    double vx;
    double vy;
} state_vector;

typedef struct {
    state_vector x;
    double p[kalman_states][kalman_states];
    double q[kalman_states][kalman_states];
    double r[2][2];
    double dt;
} kalman_filter;

typedef struct {
    double kp;
    double ki;
    double kd;
    double integral;
    double prev_error;
    double output;
} pid_controller;

typedef struct {
    int motor_a;
    int motor_b;
    int pwm;
    int dir;
} motor_ctrl;

typedef struct {
    int state;
    int prev_state;
    long transition_time;
    double mode_data[10];
} fsm_state;

typedef struct {
    gps_data gps;
    kalman_filter kf;
    pid_controller pid_x;
    pid_controller pid_y;
    motor_ctrl motor;
    fsm_state fsm;
    int socket_fd;
    int running;
    int low_power;
    pthread_mutex_t data_lock;
} robot_system;

robot_system robot = {0};

void matrix_mult(double a[kalman_states][kalman_states], double b[kalman_states][kalman_states], double c[kalman_states][kalman_states]) {
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < kalman_states; j++) {
            c[i][j] = 0;
            for(int k = 0; k < kalman_states; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void matrix_add(double a[kalman_states][kalman_states], double b[kalman_states][kalman_states], double c[kalman_states][kalman_states]) {
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < kalman_states; j++) {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

void matrix_transpose(double a[kalman_states][kalman_states], double c[kalman_states][kalman_states]) {
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < kalman_states; j++) {
            c[j][i] = a[i][j];
        }
    }
}

double matrix_determinant_2x2(double m[2][2]) {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

void matrix_inverse_2x2(double m[2][2], double inv[2][2]) {
    double det = matrix_determinant_2x2(m);
    if(det == 0) return;
    inv[0][0] = m[1][1] / det;
    inv[0][1] = -m[0][1] / det;
    inv[1][0] = -m[1][0] / det;
    inv[1][1] = m[0][0] / det;
}

void kalman_init(kalman_filter *kf, double dt) {
    kf->dt = dt;
    kf->x.x = 0;
    kf->x.y = 0;
    kf->x.vx = 0;
    kf->x.vy = 0;
    
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < kalman_states; j++) {
            kf->p[i][j] = (i == j) ? 1.0 : 0.0;
            kf->q[i][j] = (i == j) ? 0.001 : 0.0;
        }
    }
    
    kf->r[0][0] = 0.1;
    kf->r[0][1] = 0.0;
    kf->r[1][0] = 0.0;
    kf->r[1][1] = 0.1;
}

void kalman_predict(kalman_filter *kf) {
    state_vector x_pred = kf->x;
    x_pred.x += kf->x.vx * kf->dt;
    x_pred.y += kf->x.vy * kf->dt;
    double f[kalman_states][kalman_states] = {
        {1, 0, kf->dt, 0},
        {0, 1, 0, kf->dt},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    
    double ft[kalman_states][kalman_states];
    matrix_transpose(f, ft);
    double fpt[kalman_states][kalman_states];
    matrix_mult(f, kf->p, fpt);
    double p_pred[kalman_states][kalman_states];
    matrix_mult(fpt, ft, p_pred);
    matrix_add(p_pred, kf->q, kf->p);
    kf->x = x_pred;
}

void kalman_update(kalman_filter *kf, double z_x, double z_y) {
    double h[2][kalman_states] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0}
    };
    
    double ht[kalman_states][2] = {
        {1, 0},
        {0, 1},
        {0, 0},
        {0, 0}
    };
    
    double hph[2][2] = {0};
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k = 0; k < kalman_states; k++) {
                hph[i][j] += h[i][k] * kf->p[k][j];
            }
        }
    }
    
    double hpht[2][2];
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            hpht[i][j] = hph[i][j] + kf->r[i][j];
        }
    }
    
    double s_inv[2][2];
    matrix_inverse_2x2(hpht, s_inv);
    double k[kalman_states][2] = {0};
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < 2; j++) {
            for(int k_idx = 0; k_idx < 2; k_idx++) {
                k[i][j] += kf->p[i][k_idx] * h[k_idx][0] * s_inv[k_idx][j];
            }
        }
    }
    
    double y[2] = {z_x - kf->x.x, z_y - kf->x.y};
    kf->x.x += k[0][0] * y[0] + k[0][1] * y[1];
    kf->x.y += k[1][0] * y[0] + k[1][1] * y[1];
    kf->x.vx += k[2][0] * y[0] + k[2][1] * y[1];
    kf->x.vy += k[3][0] * y[0] + k[3][1] * y[1];
    double i_mat[kalman_states][kalman_states];
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < kalman_states; j++) {
            i_mat[i][j] = (i == j) ? 1.0 : 0.0;
            for(int m = 0; m < 2; m++) {
                i_mat[i][j] -= k[i][m] * h[m][j];
            }
        }
    }
    
    double new_p[kalman_states][kalman_states];
    matrix_mult(i_mat, kf->p, new_p);
    for(int i = 0; i < kalman_states; i++) {
        for(int j = 0; j < kalman_states; j++) {
            kf->p[i][j] = new_p[i][j];
        }
    }
}

void pid_init(pid_controller *pid, double kp, double ki, double kd) {
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->integral = 0;
    pid->prev_error = 0;
    pid->output = 0;
}

double pid_update(pid_controller *pid, double error, double dt) {
    pid->integral += error * dt;
    if(pid->integral > 100) pid->integral = 100;
    if(pid->integral < -100) pid->integral = -100;
    double derivative = (error - pid->prev_error) / dt;
    pid->output = pid->kp * error + pid->ki * pid->integral + pid->kd * derivative;
    if(pid->output > motor_max_pwm) pid->output = motor_max_pwm;
    if(pid->output < -motor_max_pwm) pid->output = -motor_max_pwm;
    pid->prev_error = error;
    return pid->output;
}

void motor_init(motor_ctrl *motor, int pin_a, int pin_b) {
    motor->motor_a = pin_a;
    motor->motor_b = pin_b;
    motor->pwm = 0;
    motor->dir = 0;
}

void motor_set_speed(motor_ctrl *motor, int speed) {
    motor->pwm = (speed > motor_max_pwm) ? motor_max_pwm : (speed < -motor_max_pwm) ? -motor_max_pwm : speed;
    motor->dir = (speed >= 0) ? 1 : -1;
}

void fsm_init(fsm_state *fsm) {
    fsm->state = 0;
    fsm->prev_state = -1;
    fsm->transition_time = 0;
    for(int i = 0; i < 10; i++) fsm->mode_data[i] = 0;
}

void fsm_update(fsm_state *fsm, double x, double y, double vx, double vy) {
    double dist = sqrt(x * x + y * y);
    double speed = sqrt(vx * vx + vy * vy);
    int new_state = fsm->state;
    if(dist < 1.0 && speed < 0.1) new_state = 0;
    else if(dist < 5.0 && speed < 1.0) new_state = 1;
    else if(dist < 20.0 && speed < 3.0) new_state = 2;
    else new_state = 3;
    if(speed > 5.0) new_state = 4;
    if(new_state != fsm->state) {
        fsm->prev_state = fsm->state;
        fsm->state = new_state;
        fsm->transition_time = time(NULL);
    }
}

void gps_parse(const char *nmea, gps_data *gps) {
    char *token = strtok((char*)nmea, ",");
    int field = 0;
    while(token && field < 13) {
        if(field == 2) gps->lat = atof(token) / 100.0;
        else if(field == 4) gps->lon = atof(token) / 100.0;
        else if(field == 9) gps->alt = atof(token);
        else if(field == 7) gps->spd = atof(token) * 0.51444;
        else if(field == 8) gps->hdg = atof(token);
        
        token = strtok(NULL, ",");
        field++;
    }
    gps->ts = time(NULL);
}

void* gps_thread(void *arg) {
    char buffer[gps_buffer_size];
    while(robot.running) {
        if(robot.low_power) {
            usleep(500000);
            continue;
        }
        
        fgets(buffer, sizeof(buffer), stdin);
        if(strstr(buffer, "GPRMC")) {
            pthread_mutex_lock(&robot.data_lock);
            gps_parse(buffer, &robot.gps);
            pthread_mutex_unlock(&robot.data_lock);
        }
        
        usleep(10000);
    }
    
    return NULL;
}

void* control_thread(void *arg) {
    double dt = 1.0 / sample_rate;
    while(robot.running) {
        if(robot.low_power) {
            usleep(100000);
            continue;
        }
        
        pthread_mutex_lock(&robot.data_lock);
        kalman_predict(&robot.kf);
        kalman_update(&robot.kf, robot.gps.lon, robot.gps.lat);
        double target_x = 10.0;
        double target_y = 10.0;
        double error_x = target_x - robot.kf.x.x;
        double error_y = target_y - robot.kf.x.y;
        double ctrl_x = pid_update(&robot.pid_x, error_x, dt);
        double ctrl_y = pid_update(&robot.pid_y, error_y, dt);
        double speed = sqrt(ctrl_x * ctrl_x + ctrl_y * ctrl_y);
        motor_set_speed(&robot.motor, (int)speed);
        fsm_update(&robot.fsm, robot.kf.x.x, robot.kf.x.y, robot.kf.x.vx, robot.kf.x.vy);
        pthread_mutex_unlock(&robot.data_lock);
        usleep((unsigned int)(dt * 1000000));
    }
    
    return NULL;
}

void* telemetry_thread(void *arg) {
    struct sockaddr_in server_addr;
    char buffer[tcp_buffer_size];
    robot.socket_fd = socket(af_inet, sock_stream, 0);
    if(robot.socket_fd < 0) return NULL;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = af_inet;
    server_addr.sin_port = htons(5000);
    inet_pton(af_inet, "192.168.1.100", &server_addr.sin_addr);
    if(connect(robot.socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        close(robot.socket_fd);
        return NULL;
    }
    
    while(robot.running) {
        if(robot.low_power) {
            usleep(1000000);
            continue;
        }
        
        pthread_mutex_lock(&robot.data_lock);
        snprintf(buffer, sizeof(buffer), 
            "%.6f,%.6f,%.2f,%.2f,%.2f,%d,%d\n",
            robot.kf.x.x,
            robot.kf.x.y,
            robot.kf.x.vx,
            robot.kf.x.vy,
            robot.motor.pwm,
            robot.fsm.state,
            robot.gps.ts);
        send(robot.socket_fd, buffer, strlen(buffer), 0);
        pthread_mutex_unlock(&robot.data_lock);
        sleep(1);
    }
    
    close(robot.socket_fd);
    return NULL;
}

void power_management() {
    if(robot.kf.x.vx < 0.01 && robot.kf.x.vy < 0.01) {
        robot.low_power = 1;
    } else {
        robot.low_power = 0;
    }
}

void signal_handler(int sig) {
    robot.running = 0;
}

int main() {
    robot.running = 1;
    robot.low_power = 0;
    pthread_mutex_init(&robot.data_lock, NULL);
    kalman_init(&robot.kf, 1.0 / sample_rate);
    pid_init(&robot.pid_x, 2.0, 0.5, 1.0);
    pid_init(&robot.pid_y, 2.0, 0.5, 1.0);
    motor_init(&robot.motor, 17, 27);
    fsm_init(&robot.fsm);
    signal(sigint, signal_handler);
    signal(sigterm, signal_handler);
    pthread_t gps_tid, ctrl_tid, tel_tid;
    pthread_create(&gps_tid, NULL, gps_thread, NULL);
    pthread_create(&ctrl_tid, NULL, control_thread, NULL);
    pthread_create(&tel_tid, NULL, telemetry_thread, NULL);
    while(robot.running) {
        power_management();
        sleep(1);
    }
    
    pthread_join(gps_tid, NULL);
    pthread_join(ctrl_tid, NULL);
    pthread_join(tel_tid, NULL);
    pthread_mutex_destroy(&robot.data_lock);
    return 0;
}
