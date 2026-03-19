#!/usr/bin/python3

# THIS FILE MUST NEVER BE RELEASED TO STUDENTS

import time
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from cw1_world_spawner.srv import TaskSetup, Task1Service, Task2Service, Task3Service
import numpy as np
import os

from world_spawner import Task1, Task2, Task3, World, set_context
from coursework_world_spawner_lib.coursework_world_spawner import WorldSpawner

NODE = None


def loginfo(msg):
  NODE.get_logger().info(str(msg))


def logwarn(msg):
  NODE.get_logger().warning(str(msg))


def logdebug(msg):
  NODE.get_logger().debug(str(msg))


def logerr(msg):
  NODE.get_logger().error(str(msg))


def now_sec():
  return NODE.get_clock().now().nanoseconds / 1e9


def sleep(duration):
  time.sleep(float(duration))


def create_rate(hz):
  return NODE.create_rate(float(hz))

# # hint: an easy way to 'disable' the randomness in the task spawning is:
# myseed = 0
# np.random.seed(myseed) # choose any int as your seed

# ----- key coursework task parameters ----- #

# task 1 parameters                 
T1_BOX_X_LIMS = [0.3, 0.5]            # xrange a box can spawn
T1_BOX_Y_LIMS = [-0.10, 0.11]         # yrange a box can spawn

# task 2 parameters
T2_N_BASKETS = 3                      # number of baskets to spawn
T2_BASKET_COLOUR_UNIQUE = False       # can there only be one basket of each colour
T2_BASKET_NOISE = 50e-3               # possible noise on (x, y) for basket location

# task 3 parameters
T3_N_BOXES = 4                        # number of boxes spawn
T3_N_BASKETS = 3                      # number of baskets to spawn
T3_BASKET_COLOUR_UNIQUE = True        # can there only be one basket of each colour
T3_BOX_X_LIMS = [0.35, 0.66]          # xrange a box can spawn
T3_BOX_Y_LIMS = [-0.10, 0.11]         # yrange a box can spawn
T3_BASKET_NOISE = 50e-3               # possible noise on (x, y) for basket location

# possible goal basket locations (x, y)
BASKET_LOCATIONS = [(0.59, -0.34), 
                    (0.59,  0.34),
                    (0.31, -0.34),
                    (0.31,  0.34)]

# possible spawned box colours, do not edit as rgb for models set in sdf file
BOX_COLORS = {'purple': [0.8, 0.1, 0.8],
              'red':    [0.8, 0.1, 0.1], 
              'blue':   [0.1, 0.1, 0.8]}

# possible goal basket colours, do not edit as rgb for models set in sdf file
BASKET_COLORS = {'purple': [0.8, 0.1, 0.8],
                 'red':    [0.8, 0.1, 0.1], 
                 'blue':   [0.1, 0.1, 0.8]}

# ----- variables used for marking ----- #

# time in seconds in each task before timeout
T1_TEST_TIME = 60
T2_TEST_TIME = 120
T3_TEST_TIME = 360

# marking variables
T3_MIN_TASK_DURATION = 120
T2_MAX_MARKS =100.

T3_MAX_MARK = 100

TEST_REPORT_STR = """"""



# validation method, not for students
def task1_spawn_test_objects(self, validation_scenario):
  

  self.world_spawner.despawn_all(keyword='object', exceptions='golf')
  self.world_spawner.despawn_all(keyword='basket',exceptions='golf')
  #self.goal_pt = self.world.randomize_goal([BASKET_LOCATIONS[0]])

  global TEST_REPORT_STR
  TEST_REPORT_STR += "\nTesting for task 1\n"

  # scenario 0: easy pick from right in front of the robot
  if validation_scenario == 0:
    loginfo("Creating validation scenario 0")
    TEST_REPORT_STR += "Creating validation scenario 0\n"

    np.random.seed(10)
    
    self.spawn_random_goal_baskets(num=1)

    #self.goal_pt = self.world.spawn_goal_basket(np.array(BASKET_LOCATIONS[0]).tolist())
    self.spawn_box_object( name='testobject1',
                                                    xlims = [0.4, 0.4],
                                                    ylims = [0.0, 0.0])

  # scenario 1: pick from right next to the goal box
  elif validation_scenario == 1:
    loginfo("Creating validation scenario 1")
    TEST_REPORT_STR += "Creating validation scenario 1\n"

    np.random.seed(11)

    self.spawn_random_goal_baskets(num=1)
 

    #self.goal_pt = self.world.spawn_goal_basket(np.array(BASKET_LOCATIONS[1]).tolist())
    self.spawn_box_object( name='testobject1',
                                                    xlims = [0.5, 0.5],
                                                    ylims = [-0.12, -0.12])
  
  # scenario 2: pick from top cube on stack of two
  
  elif validation_scenario == 2:
    loginfo("Creating validation scenario 2")
    TEST_REPORT_STR += "Creating validation scenario 2\n"
    np.random.seed(12)

    self.spawn_random_goal_baskets(num=1)
 

    #self.goal_pt = self.world.spawn_goal_basket(np.array(BASKET_LOCATIONS[0]).tolist())
    #self.spawn_box_object(name='testobject2',xlims = [0.3, 0.3],
    #                                        ylims = [-0.04, -0.04],
    #                                        zlims = [0.03, 0.03])
    self.spawn_box_object(name='testobject1',xlims = [0.3, 0.3],
                                            ylims = [-0.04, -0.04],
                                            zlims = [0.06, 0.06])
                                                    
  else: raise RuntimeError("validation scenario must equal 0, 1, or 2")

  return

# validation method, not for students
def task1_begin_test(self, validation_scenario):

  global TEST_REPORT_STR
  

  t1_grade = 0    
  ##### START TASK TIMER
  # logdebug("Starting Task1 timer.")
  # self.trial_timeout = False
  # rospy.Timer(rospy.Duration(T1_TEST_TIME), self.stop_callback, oneshot=True)
  ##### SEND TASK REQUEST
  sleep(1)
  start_time = now_sec()
  init_pose = self.models['testobject1'].get_model_state().pose
  init_pos_np = self.get_position_from_pose(init_pose)
  success = self.prepare_for_task_request(Task1Service, self.service_to_request)
  if success is None:
    loginfo("[RESULT - %d/100] Task1 - Fail, no service found!"%t1_grade)
    TEST_REPORT_STR += "[RESULT - %d/100] Task1 - Fail, no service found!\n"%t1_grade
    return False

  resp = self.send_task1_request(init_pose)

  goal_pos_np = np.asarray([self.basket_points[0].x,self.basket_points[0].y, self.basket_points[0].z])
 

  
  rate = create_rate(10)
  while rclpy.ok():
    ball_pose = self.models['testobject1'].get_model_state().pose
    ball_pos_np = self.get_position_from_pose(ball_pose)

    print('ball_pos_np POINTS IS',ball_pos_np)
    dx,dy,dz = np.abs(goal_pos_np - ball_pos_np)

    if dx < self.world.basket_side_length/2. and dy < self.world.basket_side_length/2.:
      t1_grade = 100
      loginfo("[RESULT - %d/100] Task1 - Successful pick and place!"%t1_grade)
      TEST_REPORT_STR += "[RESULT - %d/100] Task1 - Successful pick and place!\n"%t1_grade
      TEST_REPORT_STR += "[EXECUTION TIME] Task1 - " + str(now_sec() - start_time) + " seconds\n"      
      return True

    else: 
      d_move  = np.sqrt(np.sum(np.power(init_pos_np[:2] - ball_pos_np[:2],2)))
      d_goal  = np.sqrt(np.sum(np.power(goal_pos_np[:2] - ball_pos_np[:2],2)))
      d_total = np.sqrt(np.sum(np.power(goal_pos_np[:2] - init_pos_np[:2],2)))
      t1_grade = int( min( max( 100*(d_total-d_goal) / d_total , 80 ) , 0) )
      loginfo("[RESULT - %d/100]  Moved %0.3f, distance to goal %0.3f"%(t1_grade, d_move, d_goal))
      TEST_REPORT_STR += "[RESULT - %d/100]  Moved %0.3f, distance to goal %0.3f\n"%(t1_grade, d_move, d_goal)
      TEST_REPORT_STR += "[EXECUTION TIME] Task1 - " + str(now_sec() - start_time) + " seconds\n"
      return False

    rate.sleep()

# validation method, not for students
def task2_spawn_test_objects(self, validation_scenario):

  self.world_spawner.despawn_all(keyword='object',exceptions='golf')
  # self.world_spawner.despawn_all(keyword='basket',exceptions='golf')

  global TEST_REPORT_STR
  TEST_REPORT_STR += "\nTesting for task 2\n"

  if validation_scenario == 0:
    loginfo("Creating validation scenario 0")
    TEST_REPORT_STR += "Creating validation scenario 0\n"
    np.random.seed(13)
    
    T2_N_BASKETS = 3 
    T2_BASKET_COLOUR_UNIQUE = False      
    T2_BASKET_NOISE = 0     

    # do 1
    self.world_spawner.despawn_all(keyword='object',exceptions='golf')
    self.spawn_random_goal_baskets(num=T2_N_BASKETS, unique_colours=T2_BASKET_COLOUR_UNIQUE,
                                   noise=T2_BASKET_NOISE, save_empty=True)


  if validation_scenario == 1:
    loginfo("Creating validation scenario 1")
    TEST_REPORT_STR += "Creating validation scenario 1\n"
    np.random.seed(5)
    # do 2
    T2_N_BASKETS = 1
    T2_BASKET_COLOUR_UNIQUE = False
    T2_BASKET_NOISE = 0. 
    
    self.world_spawner.despawn_all(keyword='object',exceptions='golf')
    self.spawn_random_goal_baskets(num=T2_N_BASKETS, unique_colours=T2_BASKET_COLOUR_UNIQUE,
                                   noise=T2_BASKET_NOISE, save_empty=True)


  if validation_scenario == 2:
    loginfo("Creating validation scenario 2")
    TEST_REPORT_STR += "Creating validation scenario 2\n"

    np.random.seed(65)
    # do 3
    T2_N_BASKETS = 3
    T2_BASKET_COLOUR_UNIQUE = False      
    T2_BASKET_NOISE = 0.05

    self.world_spawner.despawn_all(keyword='object',exceptions='golf')
    self.spawn_random_goal_baskets(num=T2_N_BASKETS, unique_colours=T2_BASKET_COLOUR_UNIQUE,
                                   noise=T2_BASKET_NOISE, save_empty=True)



  return  

# validation method, not for students
def task2_begin_test(self, validation_scenario):

  global TEST_REPORT_STR
  
  t2_grade = 0
  ##### START TASK TIMER
  # logdebug("Starting Task2 timer.")
  # self.trial_timeout = False
  # rospy.Timer(rospy.Duration(T2_TEST_TIME), self.stop_callback, oneshot=True)
  
  start_time = now_sec()
  ##### SEND TASK REQUEST
  success = self.prepare_for_task_request(Task2Service, self.service_to_request)
  
  if success is None:
    loginfo("[RESULT - %d/100] Task2 - Fail, no service found!" % t2_grade )
    TEST_REPORT_STR += "[RESULT - %d/100] Task2 - Fail, no service found!\n" % t2_grade
    return t2_grade

  
  basket_goal_colours = self.basket_colours

  resp = self.send_task2_request()

  basket_returned_colours =  resp.basket_colours

  N_GOALS = len(basket_goal_colours)
  N_ESTIMATES = len(basket_returned_colours)
  d = abs( N_GOALS - N_ESTIMATES )

  if len(basket_returned_colours) != 0 :

    if N_ESTIMATES==0:
      t2_grade = 0 
      loginfo("[RESULT - %d/100] Task2 - No colour detected"% ( 
                      int(t2_grade)))
      TEST_REPORT_STR += "[RESULT - %d/100] Task2 - No colour detected\n"% ( 
                      int(t2_grade))
    

    #ensure there is no penalty for 'None' or 'empty 
    for estimate in range(N_ESTIMATES):
      if basket_returned_colours[estimate].lower() == 'none':
            basket_returned_colours[estimate] = 'empty'

    print('"""""""""""""""COLOURS""""""""""""""""""')
    print('TRUE COLOURS',basket_goal_colours )
    print('RETURNED COLOURS', basket_returned_colours )
    sorted_goal_colour = sorted(basket_goal_colours)
    sorted_returned_colour = sorted(basket_returned_colours)

    # store values 
    sorted_rightly_identified = 0 
    rightly_identified = 0 
    right_colour = 0 


    #TEST 1: check if colour is identified if N_ESTIMATES < N_GOALS

    for color in basket_goal_colours:
      if color in basket_returned_colours:
          right_colour+=1
    print('right colour', right_colour)
     
    for colour in range(N_GOALS):   

        #TEST 2: check if correct number of identified colours
        #if sorted_goal_colour[colour] == sorted_returned_colour[colour]:
        #   sorted_rightly_identified += 1
	
      if N_ESTIMATES == N_GOALS:
      #Test 3: right colour and position 
        if basket_goal_colours[colour] == basket_returned_colours[colour]:
              rightly_identified += 1

      elif N_ESTIMATES > N_GOALS: 
        if basket_goal_colours[colour] == basket_returned_colours[colour:N_ESTIMATES]:
                rightly_identified += 1
      
      else: 
        if basket_goal_colours[colour:N_ESTIMATES] == basket_returned_colours[colour:N_ESTIMATES]:
                rightly_identified += 1

      

    if N_ESTIMATES > N_GOALS:
      ## Give marks for correct number of returned string 
      t2_grade = T2_MAX_MARKS * 0.25 *(1 - abs(N_GOALS-N_ESTIMATES)/N_ESTIMATES)
      loginfo("[RESULT - %d/100] Task2 - Correct number of string elements  %d/%d"% ( 
                    int(t2_grade), N_ESTIMATES, N_GOALS))
      TEST_REPORT_STR += "[RESULT - %d/100] Task2 - Correct number of string elements %d/%d\n"% ( 
                    int(t2_grade), N_ESTIMATES, N_GOALS)
    
    if N_ESTIMATES <= N_GOALS:

      ## Give marks for correct number of returned string 
      t2_grade = T2_MAX_MARKS * 0.25 *(1 - abs(N_GOALS-N_ESTIMATES)/N_GOALS)
      loginfo("[RESULT - %d/100] Task2 - Correct number of string elements  %d/%d"% ( 
                    int(t2_grade), N_ESTIMATES, N_GOALS))
      TEST_REPORT_STR += "[RESULT - %d/100] Task2 - Correct number of string elements %d/%d\n"% ( 
                    int(t2_grade), N_ESTIMATES, N_GOALS)
    
    ## Give marks for correct indedified colours in the returned string
    t2_grade += T2_MAX_MARKS * 0.25 *(1 - abs(N_ESTIMATES-right_colour)/N_ESTIMATES)
    loginfo("[RESULT - %d/100] Task2 - Number of correctly identified colours %d/%d"% ( 
                  int(t2_grade), right_colour, N_ESTIMATES))
    TEST_REPORT_STR += "[RESULT - %d/100] Task2 - Number of correctly identified colours %d/%d\n"% ( 
                  int(t2_grade), right_colour, N_ESTIMATES)
    
    ## Give Marks for correctly identifying colours and puthing them in the right order 
    t2_grade += T2_MAX_MARKS* 0.5 *(1 - abs(N_GOALS-rightly_identified)/N_GOALS)
    loginfo("[RESULT - %d/100] Task2 - TOTAL: Correctly Identified Colours and Order %d/%d"% ( 
                    int(t2_grade), rightly_identified, N_GOALS))
    TEST_REPORT_STR += "[RESULT - %d/100] Task2 - TOTAL:  Correctly Identified Colours and Order %d/%d\n"% ( 
                    int(t2_grade), rightly_identified, N_GOALS)
    

  else:
    t2_grade = 0 
    loginfo("[RESULT - %d/100] Task2 - No colour detected"% ( 
                    int(t2_grade)))
    TEST_REPORT_STR += "[RESULT - %d/100] Task2 - No colour detected\n"% ( 
                    int(t2_grade))
  TEST_REPORT_STR += "[EXECUTION TIME] Task2 - " + str(now_sec() - start_time) + " seconds\n"    
  return

# validation method, not for students
def task3_spawn_test_objects(self, validation_scenario):
  
  global TEST_REPORT_STR
  TEST_REPORT_STR += "\nStarting test for task 3\n"
  self.world_spawner.despawn_all(keyword='object',exceptions='golf')
  self.world_spawner.despawn_all(keyword='basket',exceptions='golf')

  np.random.seed(23)
 
  global T3_N_BOXES
  global BOX_COLOR
  global T3_N_BASKETS

  

  if validation_scenario==0:

    np.random.seed(11)


    T3_N_BASKETS = 4
    T3_N_BOXES = 3
    T3_BASKET_COLOUR_UNIQUE = True
    T3_BASKET_NOISE = 0.
    BOX_COLOR = ['purple','red','blue']
    self.basket_points = []
    self.basket_colours = []

    for basket in range(T3_N_BASKETS): 
      colour = list(BASKET_COLORS.keys())[np.random.randint(0, len(BASKET_COLORS.keys()))]
      name='basket'+str(basket)
      goal_basket_pos = self.world.spawn_goal_basket(BASKET_LOCATIONS[basket],colour=colour, name=name)
     
      self.basket_points.append(goal_basket_pos)
      self.basket_colours.append(colour)

    self.box_locs = []
    ys = np.arange(T3_BOX_Y_LIMS[0], T3_BOX_Y_LIMS[1], step=self.world.tile_side_length)
    xs = np.arange(T3_BOX_X_LIMS[0], T3_BOX_X_LIMS[1], step=self.world.tile_side_length)
    for i, x_i in enumerate(xs):
      for j, y_j in enumerate(ys):
        if (np.abs(x_i) < self.world.robot_safety_radius and 
            np.abs(y_j) < self.world.robot_safety_radius): 
          continue
        self.box_locs.append([x_i, y_j])

    # spawn new objects randomly within limits
    np.random.shuffle(self.box_locs)
    self.box_locs = self.box_locs[:T3_N_BOXES]
    for i,loc in enumerate(self.box_locs):
      box_colour = BOX_COLOR[i]
      mname = 'test_object_'+str(i)
      self.spawn_box_object(name=mname, color=box_colour,
                              xlims=[loc[0],loc[0]], ylims=[loc[1],loc[1]])


  if validation_scenario==1:
    
    T3_N_BASKETS = 2
    T3_N_BOXES = 7
  
    BOX_COLOR = ['purple','red','red','blue','blue','blue','blue']

    self.basket_points = []
    self.basket_colours = []
    
    #self.world_spawner.despawn_all()
    self.world_spawner.despawn_all(keyword='object',exceptions='golf')
    self.world_spawner.despawn_all(keyword='basket',exceptions='golf')

    colour = ["purple","red"]
    for basket in range(T3_N_BASKETS): 
      name='basket'+str(basket)
      goal_basket_pos = self.world.spawn_goal_basket(BASKET_LOCATIONS[basket+1],colour=colour[basket],name=name)
     
      self.basket_points.append(goal_basket_pos)

    self.basket_colours=colour
    
     
    self.box_locs = []
    ys = np.arange(T3_BOX_Y_LIMS[0], T3_BOX_Y_LIMS[1], step=self.world.tile_side_length)
    xs = np.arange(T3_BOX_X_LIMS[0], T3_BOX_X_LIMS[1], step=self.world.tile_side_length)
    for i, x_i in enumerate(xs):
      for j, y_j in enumerate(ys):
        if (np.abs(x_i) < self.world.robot_safety_radius and 
            np.abs(y_j) < self.world.robot_safety_radius): 
          continue
        self.box_locs.append([x_i, y_j])

    # spawn new objects randomly within limits
    np.random.shuffle(self.box_locs)
    loginfo('before%d'%len(self.box_locs))
    

    self.box_locs = self.box_locs[:T3_N_BOXES]
    
    loginfo('after%d'%len(self.box_locs))

    for i,loc in enumerate(self.box_locs):
      box_colour = BOX_COLOR[i]
      mname = 'test_object_'+str(i)
      self.spawn_box_object(name=mname, color=box_colour,
                              xlims=[loc[0],loc[0]], ylims=[loc[1],loc[1]])
    

  if validation_scenario==2: 

    T3_N_BASKETS = 2
    T3_N_BOXES = 5
    BOX_COLOR = ['red'] * 5
    self.basket_points = []
    self.basket_colours = []
    colour = ["red"]*2
    
    #self.world_spawner.despawn_all()
    self.world_spawner.despawn_all(keyword='object',exceptions='golf')
    self.world_spawner.despawn_all(keyword='basket',exceptions='golf')

    for basket in range(T3_N_BASKETS): 
      name='basket'+str(basket)
      goal_basket_pos = self.world.spawn_goal_basket(BASKET_LOCATIONS[basket],colour=colour[basket], name=name)
      
      self.basket_points.append(goal_basket_pos)
    self.basket_colours=colour

    self.box_locs = []
    ys = np.arange(T3_BOX_Y_LIMS[0], T3_BOX_Y_LIMS[1], step=self.world.tile_side_length)
    xs = np.arange(T3_BOX_X_LIMS[0], T3_BOX_X_LIMS[1], step=self.world.tile_side_length)
    for i, x_i in enumerate(xs):
      for j, y_j in enumerate(ys):
        if (np.abs(x_i) < self.world.robot_safety_radius and 
            np.abs(y_j) < self.world.robot_safety_radius): 
          continue
        self.box_locs.append([x_i, y_j])

    # spawn new objects randomly within limits
    np.random.shuffle(self.box_locs)
    self.box_locs = self.box_locs[:T3_N_BOXES]
    for i,loc in enumerate(self.box_locs):
      box_colour = BOX_COLOR[i]
      mname = 'test_object_'+str(i)
      self.spawn_box_object(name=mname, color=box_colour,
                              xlims=[loc[0],loc[0]], ylims=[loc[1],loc[1]])  
  return 

# validation method, not for students
def task3_begin_test(self, validation_scenario):

  global TEST_REPORT_STR
  

  t3_grade = 0
  
 
  ##### SEND TASK REQUEST
  import time
  T3_start_time = time.time()
  success = self.prepare_for_task_request(Task3Service, self.service_to_request)
  
  if success is None:
    loginfo("[RESULT - %d/100] Task3 - Fail, no service found!"%(int(t3_grade)))
    TEST_REPORT_STR += "[RESULT - %d/100] Task3 - Fail, no service found!\n"%(int(t3_grade))
    return t3_grade

  resp = self.send_task3_request()
  # rate = create_rate(10)
  # while(not not rclpy.ok()):
  n_boxes_in_basket = 0

  ### BOXES: Take test data of boxes from simulaiton
  test_box_colours = []
  test_box_pos = []

  #save the data from spawned obxes in simualtion 

  for box in range(T3_N_BOXES):
    name ='test_object_'+str(box) 
    box_colour = self.models[name].mdict["model_name"].split("_")[1]
    box_pos = self.get_position_from_pose(self.models[name].get_model_state().pose)

    test_box_colours.append(box_colour)
    test_box_pos.append(box_pos)
  
  goal_box_colours = np.array(self.basket_colours)
  #if goal_box_colours.shape[1] > 1: goal_box_colours = goal_box_colours[0]


  print('"""""""""""""""COLOURS AND POSITION """"""""""""""""""')
  print('BOX COLOURS',test_box_colours)
  # print('BOX POSITION', test_box_pos)
  # print()
  print('BASKET COLOURS',goal_box_colours)
  # print('BASKET POSITION', self.basket_points)


  
  #check if boxes in the baskets 
  n_boxes_in_basket = 0 
  right_colour = 0
  right_colour2 =0
  which_box = []
  which_basket = []



  # how many boxes of basket colour
  for color in test_box_colours:
    if color in goal_box_colours:
        right_colour+=1
  print('right colour', right_colour)
  

  for box in range(T3_N_BOXES):
    for basket in range(T3_N_BASKETS):
      test_pos = test_box_pos[box]
      goal_pos = np.asarray([self.basket_points[basket].x, self.basket_points[basket].y, self.basket_points[basket].z])
      dx,dy,dz = np.abs(goal_pos - test_pos)
      print("CHECKKK", dx) 
      if dx < self.world.basket_side_length/2. and dy < self.world.basket_side_length/2.:
        n_boxes_in_basket+=1
        print("num of boxes", n_boxes_in_basket) 
        which_box.append(box)
        which_basket.append(basket)
  
  #check if the box inside bask is the correct colour
  
  correct_colour = 0 
  for box_in in range(n_boxes_in_basket):
      check_box = which_box[box_in]
      check_basket = which_basket[box_in]
      if test_box_colours[check_box] == goal_box_colours[check_basket]:
          correct_colour +=1

   ## Give marks for correct indedified colours in the returned string
  # t3_grade += T3_MAX_MARK * 0.25 *(1 - abs(len(test_box_colours)-right_colour)/len(test_box_colours))
  # loginfo("[RESULT - %d/100] Task3 - Number of correctly identified colours %d/%d"% ( 
  #               int(len(test_box_colours)), right_colour, len(test_box_colours)))
  # TEST_REPORT_STR += "[RESULT - %d/100] Task3 - Number of correctly identified colours %d/%d\n"% ( 
  #               int(len(test_box_colours)), right_colour, len(test_box_colours))
  

  if right_colour != 0:
    t3_grade+=T3_MAX_MARK *0.5*(1 - abs(right_colour - n_boxes_in_basket)/right_colour)
  else: 
    t3_grade +=0
  loginfo("[RESULT - %d/100] Task3 - There are %d/%d boxes inside the basket\n"%(int(t3_grade),n_boxes_in_basket,right_colour))
  TEST_REPORT_STR += "[RESULT - %d/100] Task3 - There are %d/%d boxes inside the basket\n"%(int(t3_grade),n_boxes_in_basket,right_colour)



  if n_boxes_in_basket != 0:
    t3_grade +=T3_MAX_MARK *0.5*(1 - abs(n_boxes_in_basket-correct_colour)/n_boxes_in_basket)
  else:
    t3_grade += 0 
  loginfo("[RESULT - %d/100] Task3 - There are %d correct boxes inside the goal basket\n"%(int(t3_grade),n_boxes_in_basket))
  TEST_REPORT_STR += "[RESULT - %d/100] Task3 - There are %d correct boxes inside the goal basket\n"%(int(t3_grade),n_boxes_in_basket)

  TEST_REPORT_STR += "[EXECUTION TIME] Task3 - " + str(time.time() - T3_start_time) + " seconds\n"
  return

  raise NotImplementedError()

# apply these functions to the base class
Task1.spawn_test_objects = task1_spawn_test_objects
Task1.begin_test = task1_begin_test
Task2.spawn_test_objects = task2_spawn_test_objects
Task2.begin_test = task2_begin_test
Task3.spawn_test_objects = task3_spawn_test_objects
Task3.begin_test = task3_begin_test

def run_task1_validation():
  # does NOT wipe the test report string
  global TEST_REPORT_STR
  TEST_REPORT_STR += """Test report for task 1:\n\n"""
  loginfo("Running task 1 validation battery")
  Task1(mode='validation', validation_scenario=0)
  Task1(mode='validation', validation_scenario=1)
  Task1(mode='validation', validation_scenario=2)
  TEST_REPORT_STR += "\nEnd of test"
  print(TEST_REPORT_STR)
  save_test_report()

def run_task2_validation():
  # does NOT wipe the test report string
  global TEST_REPORT_STR
  TEST_REPORT_STR += """Test report for task 2:\n\n"""
  loginfo("Running task 2 validation battery")
  Task2(mode='validation', validation_scenario=0)
  Task2(mode='validation', validation_scenario=1)
  Task2(mode='validation', validation_scenario=2)
  TEST_REPORT_STR += "\nEnd of test"
  print(TEST_REPORT_STR)
  save_test_report()

def run_task3_validation():
  # does NOT wipe the test report string
  global TEST_REPORT_STR
  TEST_REPORT_STR += """Test report for task 3:\n\n"""
  loginfo("Running task 3 validation battery")
  Task3(mode='validation', validation_scenario=0)
  Task3(mode='validation', validation_scenario=1)
  Task3(mode='validation', validation_scenario=2)
  TEST_REPORT_STR += "\nEnd of test"
  print(TEST_REPORT_STR)
  save_test_report()

def run_full_validation():
  loginfo("Running full coursework validation!")
  # wipe the test report string
  global TEST_REPORT_STR
  TEST_REPORT_STR = """Test report for all three tasks:\n\n"""
  # run tasks
  loginfo("Running task 1 validation battery")
  Task1(mode='validation', validation_scenario=0)
  Task1(mode='validation', validation_scenario=1)
  Task1(mode='validation', validation_scenario=2)
  loginfo("Running task 2 validation battery")
  Task2(mode='validation', validation_scenario=0)
  Task2(mode='validation', validation_scenario=1)
  Task2(mode='validation', validation_scenario=2)
  loginfo("Running task 3 validation battery")
  Task3(mode='validation', validation_scenario=0)
  Task3(mode='validation', validation_scenario=1)
  Task3(mode='validation', validation_scenario=2)
  # print and save test report string
  TEST_REPORT_STR += "\nEnd of test"
  print(TEST_REPORT_STR)
  save_test_report()

def save_test_report():

  # get a timestamp
  from datetime import datetime
  datestr = "%d-%m-%y-%H:%M"
  timestamp_str = datetime.now().strftime(datestr)

  path_to_pkg = get_package_share_directory("cw1_world_spawner")
  savepath = os.path.abspath(os.path.join(path_to_pkg, "..", "..", "..", "test_reports"))

  # create the fodler if it doesn't exist
  if not os.path.exists(savepath):
    os.makedirs(savepath)
    print(f"Created the test_reports folder at {savepath}")

  filename = os.path.join(savepath, f"test_report_{timestamp_str}.txt")

  with open(filename, 'w') as openfile:
        openfile.write(TEST_REPORT_STR)
        loginfo("Saved test report at: {}".format(filename))

def handle_task_request(req, response):

  loginfo("started handle_task_request")

  # Callback for selecting which task to start
  if req.task_index == 1:
    Task1(mode="coursework")
  elif req.task_index == 2:
    Task2(mode="coursework")
  elif req.task_index == 3:
    Task3(mode="coursework")
  else:
    logwarn("Unrecognized task requested")

  loginfo("finished handle_task_request")

  return response

def handle_test_request(req, response):

  loginfo("enter handle_task_request")

  # Callback for selecting which task to start
  if req.task_index == 1:
    Task1(mode="validation")
  elif req.task_index == 2:
    Task2(mode="validation")
  elif req.task_index == 3:
    Task3(mode="validation")

  # testing methods, not for students
  elif req.task_index == 10:
    run_full_validation()

  elif req.task_index == 11:
    run_task1_validation()

  elif req.task_index == 111:
    Task1(mode='validation', validation_scenario=2)

  elif req.task_index == 12:
    run_task2_validation()
    
  elif req.task_index == 13:
    run_task3_validation()
  # end not for students

  else:
    logwarn("Unrecognized task requested")

  loginfo("at end of handle_task_request")

  return response
 



if __name__ == "__main__":
  rclpy.init()
  node = Node('coursework1_test_wrapper')
  NODE = node
  world_spawner = WorldSpawner(node)
  world = World(node, world_spawner)
  set_context(node, world_spawner, world)

  node.create_service(TaskSetup, '/task', handle_task_request)
  loginfo("Ready to initiate task.")
  loginfo("Use ros2 service call /task cw1_world_spawner/srv/TaskSetup '{task_index: <INDEX>}' to start a task")

  # START TEST SERVICE, not for students
  node.create_service(TaskSetup, '/test', handle_test_request)
  loginfo("Use ros2 service call /test cw1_world_spawner/srv/TaskSetup '{task_index: <INDEX>}' to start a test")

  rclpy.spin(node)
  world_spawner.destroy()
  node.destroy_node()
  rclpy.shutdown()



# THIS FILE MUST NEVER BE RELEASED TO STUDENTS
