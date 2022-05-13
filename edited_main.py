from VideoCap import Camera
import cv2
import numpy as np
from numpy import random as np_random
import argparse
import time
import cv2
from cv2 import cuda as cv_cuda
import os
from threading import Thread
from datetime import datetime
from fpdf import FPDF


seconds_detection = 2   # put 0.05 seconds for each frame detetion

pdf = FPDF()  
pdf.set_font("Arial", size = 15)


print("[INFO] loading YOLO from disk....")
image_path='download.jpg'
yolo_labelsPath='coco.names'
yolo_weightsPath='yolov4.weights'
yolo_configPath='yolov4.cfg'

#Load the coco class labels and YOLO weights and model config
labelsPath = yolo_labelsPath # load class names
weightsPath = yolo_weightsPath
configPath = yolo_configPath
CONFIDENCE = 0.5
THRESOLD = 0.3

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
LABELS = open(labelsPath).read().strip().split("\n")

data_cam1 = {}
data_cam2 = {}
last_violation_count_cam1 = 0
no_violation_count_cam1 = 0
last_violation_count_cam2 = 0 
no_violation_count_cam2 = 0
data_cam1['overlapped_count'] = [0]
data_cam1['overlapped_time'] = [""]
data_cam2['overlapped_count'] = [0]
data_cam2['overlapped_time'] = [""]

def convert_list_to_string(lst):
	string_output = ""
	for item in lst:
		string_output += str(item) + ","
	
	# remove last comma
	string_output = string_output[:-1]

	return string_output



def check_box_overlaps(cam_obj, boxes, image):
	global no_violation_count_cam1, no_violation_count_cam2, last_violation_count_cam1, last_violation_count_cam2
	rects_to_check = []
	current_time =""
	
	now = datetime.now()
	current_time = now.strftime("%H:%M:%S")
	current_time_file = now.strftime("%H_%M_%S")
	
			

	for box in boxes:
		x,y,w,h = box[0], box[1], box[2], box[3]
		pt_tl = (x,y)
		pt_br = (x+w,y+h)
		rect = [pt_tl,pt_br]
		rects_to_check.append(rect)
	
	# print("rects_to_check",rects_to_check)
	cam_obj.check_box_overlap(rects_to_check)


	if len(cam_obj.overlapped_rect) ==0:
		## means no overap
		if cam_obj.cam_id==0:
			no_violation_count_cam1 +=1
			
		elif cam_obj.cam_id==1:
			no_violation_count_cam2 +=1
	

	if no_violation_count_cam1 >5 :
		no_violation_count_cam1 =0 # Reset to zero
		last_violation_count_cam1= 0

	if no_violation_count_cam2 >5:
		no_violation_count_cam2 =0 # Reset to zero
		last_violation_count_cam1= 0  

	if len(cam_obj.overlapped_rect) > 0:	
		detected_viol_count = len(cam_obj.overlapped_rect)
		if cam_obj.cam_id==0:	
			if detected_viol_count > last_violation_count_cam1:
				current_list = data_cam1['overlapped_count']
				current_list.append(detected_viol_count)
				current_time_list = data_cam1['overlapped_time'] 
				current_time_list.append(current_time)

				data_cam1['overlapped_count'] = current_list
				data_cam1['overlapped_time'] = current_time_list
				
				last_violation_count_cam1 = detected_viol_count

				cv2.imwrite(f'violation_screenshots/cam1_{current_time_file}.png',image)
				
		elif cam_obj.cam_id==1:
			if detected_viol_count > last_violation_count_cam2:
				current_list = data_cam2['overlapped_count']
				current_list.append(detected_viol_count)
				current_time_list = data_cam2['overlapped_time'] 
				current_time_list.append(current_time)
				
				data_cam2['overlapped_count'] = current_list
				data_cam2['overlapped_time'] = current_time_list

				last_violation_count_cam2 = detected_viol_count
				cv2.imwrite(f'violation_screenshots/cam2_{current_time_file}.png',image)

	

def nn_check(image, frame_to_detect, a):	
	(H,W) = frame_to_detect.shape[:2]	
	np_random.seed(123)
	COLORS = np_random.randint(0,255, size=(len(LABELS),3), dtype="uint8")
	ln = net.getLayerNames()
	# ln = [ln[i[0] -1] for i in net.getUnconnectedOutLayers() ]
	ln = [ln[i-1] for i in net.getUnconnectedOutLayers() ]

	blob = cv2.dnn.blobFromImage(frame_to_detect, 1/255.0, (416,416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end =time.time()
	print("[INFO] YOLO took {:.4f} FPS".format(end-start))


	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
			for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					if confidence > CONFIDENCE:
							box = detection[0:4] * np.array([W, H, W, H])
							(centerX, centerY, width, height) = box.astype("int")

							x = int(centerX - (width / 2))
							y = int(centerY - (height / 2))

							boxes.append([x, y, int(width), int(height)])
							confidences.append(float(confidence))
							classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESOLD)
	texts = list()
	boxes_to_check = []
	if len(idxs) > 0:
			for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					rect = [x,y,w,h]
					boxes_to_check.append(rect)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					if LABELS[classIDs[i]] == 'worker':
							texts.append(LABELS[classIDs[i]])   
			cv2.putText(image, "Number of workers: "+str(len(texts)), (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255, 0, 0), 2, cv2.LINE_AA) 
	print(len(texts))
	n_idle = len(texts) - len(a)
	cv2.putText(image, "Number of idle workers: "+str(n_idle), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7 , (255, 0, 0), 2, cv2.LINE_AA)
	texts1 = list()
	a = []  

	
	texts1 = list()
	texts2 = list()
	if len(idxs) > 0:
			for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					color = [int(c) for c in COLORS[classIDs[i]]]
					#cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					if LABELS[classIDs[i]] == 'worker':
							texts.append(LABELS[classIDs[i]])
					if LABELS[classIDs[i]] == 'loaded_machine':
							texts2.append(LABELS[classIDs[i]]) 
					if LABELS[classIDs[i]] == 'unloaded_machine':
							texts1.append(LABELS[classIDs[i]])   
			cv2.putText(image, "Number of loaded machines: "+str(len(texts2)), (50,125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA) 
			cv2.putText(image, "Number of unloaded machines: "+str(len(texts1)), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
	print(len(texts2))
	print(len(texts1))
	print('texts1',texts1)
	count_loaded_Machines = len(texts2)
	count_unloaded_Machines = len(texts1)
	count_workers = len(texts)
	all_counts_nn = [count_workers, n_idle, count_loaded_Machines, count_unloaded_Machines]
	return image, all_counts_nn, boxes_to_check


def main():
	save_data = False
	path_cam1 = "IMG_7278.MOV"  
	# path_cam1 = 0  
	path_cam2 = "vid2.mp4"
	# path_cam2 = 1

	cam_obj_list = []

	## Initiate camera 1 
	cam1 = Camera(cam_id=0,src=path_cam1)
	fps_cam1 = cam1.stream.get(cv2.CAP_PROP_FPS)
	n_frame_detect = int(fps_cam1 * seconds_detection)
	print(f"Detection will be after each {n_frame_detect} frames i.e. {seconds_detection} seconds....")

	cam1.start_stream()

	## Initiate camera 2 	
	cam2 = Camera(cam_id=1,src=path_cam2)
	cam2.start_stream()

	fr1_prev = cam1.frame
	fr2_prev = cam2.frame	
	# fr1_prev = cv2.resize(fr1_prev, (853,480))
	fr1_prev = cv2.resize(fr1_prev, (360,540))
	fr2_prev = cv2.resize(fr2_prev, (360,540))

	cam_obj_list.append(cam1)
	cam_obj_list.append(cam2)

	while True:

		fr1 = cam1.frame
		# fr1 = cv2.resize(fr1, (853,480))
		fr1 = cv2.resize(fr1, (360,540))
		image1 = fr1.copy()		
		fr2 = cam2.frame
		fr2 = cv2.resize(fr2, (360,540))

		cam1.get()
		cam2.get()
		if cam1.frameNo==0:			
			continue
		if cam2.frameNo==0:
			continue

		########################  CAMERA 1 PROCESSING ########################

		## 1. Check for moving objects 		
		cam1.a = []
		img1 = cam1.get_moving_objects(prev_frame=fr1_prev)

		
		## 2. Pass through NN model for detection after n seconds		
		if cam1.frameNo % n_frame_detect ==0:				
			## detection NN
			img1, all_counts_nn,  boxes = nn_check(img1, fr1, cam1.a)			

			## Appending the counts to each respective lists of cam1			
			n_working_workers = len(cam1.moving_rects)

			cam1.update_all_counts_lists(n_worker=all_counts_nn[0],n_working_worker=n_working_workers,\
				n_idle_worker=all_counts_nn[1],n_mac_load=all_counts_nn[2],n_mac_unload=all_counts_nn[3])

			
			## 3. Check Box overlaps ( machine and men )
			check_box_overlaps(cam_obj=cam1,boxes=boxes,image=img1)
		
		## 4. if overlap create screenshot and record time of overlap
		# current_time =""
		# if 'overlapped_count' in data_cam1:
		# 	if data_cam1['overlapped_count']>0:
		# 		now = datetime.now()
		# 		current_time = now.strftime("%H:%M:%S")
		# 		current_time_file = now.strftime("%H_%M_%S")
				
		# 		cv2.imwrite(f'violation_screenshots/cam1_{current_time_file}.png',img1)

		


		########################  CAMERA 2 PROCESSING ########################
		
		## 1. Check for moving objects 	
		cam2.a = []
		img2 = cam2.get_moving_objects(prev_frame=fr2_prev)		
		# print("Person entered restricted area cam 2 : ",len(cam2.overlapped_rect))
		
		## 2. Pass through NN model for detection after n seconds
		if cam2.frameNo % n_frame_detect ==0:				
			## detection NN
			img2, all_counts_nn,  boxes = nn_check(img2, fr2, cam2.a)			

			## Appending the counts to each respective lists of cam1			
			n_working_workers = len(cam2.moving_rects)

			cam2.update_all_counts_lists(n_worker=all_counts_nn[0],n_working_worker=n_working_workers,\
				n_idle_worker=all_counts_nn[1],n_mac_load=all_counts_nn[2],n_mac_unload=all_counts_nn[3])

			## 3. Check Box overlaps ( machine and men )
			check_box_overlaps(cam_obj=cam2,boxes=boxes, image=img2)

		

		## 4. if overlap create screenshot and record time of overlap
		# current_time =""
		# if 'overlapped_count' in data_cam2:		
		# 	if data_cam2['overlapped_count']>0:
		# 		now = datetime.now()
		# 		current_time = now.strftime("%H:%M:%S")
		# 		current_time_file = now.strftime("%H_%M_%S")				
		# 		cv2.imwrite(f'violation_screenshots/cam2_{current_time_file}.png',img2)

		


		############################################################################
		## 5. Save data to txt file...

		if save_data:
			# Add a page

			print("Saving the report....................")
			pdf.add_page()
			with open('data.txt','w+') as f:

				for cam_obj in cam_obj_list:
					if cam_obj.cam_id==0:						
						cam_name = "Cam1"	
					elif cam_obj.cam_id==1:						
						cam_name = "Cam2"	

					header = f"################ {cam_name} details ############################\n"
					f.writelines(header)

					n_worker_string = convert_list_to_string(cam_obj.list_count_worker)
					n_work_working_string = convert_list_to_string(cam_obj.list_count_working_workers)
					n_work_idle_string = convert_list_to_string(cam_obj.list_count_idle_workers)
					n_mach_loaded_string = convert_list_to_string(cam_obj.list_count_machines_loaded)
					n_mach_unloaded_string = convert_list_to_string(cam_obj.list_count_machines_unloaded)
					
					line = ""
					line += "No. of workers "+n_worker_string +"\n"
					line += "No. of working workers "+n_work_working_string+"\n"
					line += "No. of idle workers "+n_work_idle_string+"\n"
					line += "No. of loaded machines "+n_mach_loaded_string+"\n"
					line += "No. of unloaded machines "+n_mach_unloaded_string+"\n"					
					line += "\n"

					if cam_obj.cam_id==0:
						data_overlap = data_cam1
					elif cam_obj.cam_id==1:
						data_overlap = data_cam2
					print('data_overlap',data_overlap)
					i=1
					count_overlap_list = data_overlap['overlapped_count']
					overlap_time_list = data_overlap['overlapped_time']
			
					for i in range(len(count_overlap_list)):
						if i > 0:
							line += "No. of overlaps : "+ str(count_overlap_list[i]) + " - at time " + str(overlap_time_list[i]) + "\n"
						


					line += "\n"
					
					f.writelines(line)
					

			f.close()

			with open('data.txt','r') as f:
				for x in f:
					pdf.cell(200, 10, txt = x, ln = 1, align = 'L')


			save_data = False

			pdf.output("report.pdf")   
		# print("Person entered restricted area cam 1 : ",len(cam1.overlapped_rect))		







		## Reset All the stored points here
		cam1.overlapped_rect = []
		cam1.moving_rects = []
		cam2.overlapped_rect = []
		cam2.moving_rects = []

		# Assign current frame to previous frame for subtraction
		fr1_prev = fr1 
		fr2_prev = fr2

		# Display the streams
		# cv2.namedWindow("frame1",cv2.WINDOW_NORMAL)
		# cv2.namedWindow("frame2",cv2.WINDOW_NORMAL)

		width1 = int(fr1_prev.shape[1])
		height1 = int(fr1_prev.shape[0])
		print(height1,' ',width1)
		width2 = int(fr2_prev.shape[1])
		height2 = int(fr2_prev.shape[0])
		# frame = cv2.resize(frame, (1080,600))
		frame = np.zeros((720,1080,3), np.uint8)
		smaller_frame1 = cv2.resize(img1, (0,0), fx=1,fy=1)
		smaller_frame2 = cv2.resize(img2, (0,0), fx=1,fy=1)
		frame[:int(height), :int(width1)] = smaller_frame1
		frame[int(height2):, :int(width2)] = smaller_frame2


		cv2.imshow("frame1",frame)
		# cv2.imshow('frame2',img2)

		key = cv2.waitKey(1)
		if key == ord('q'):
			cam1.stop()
			cam2.stop()
			break

		if key == ord('p'):
			cv2.waitKey(-1)
		
		if key ==ord('s'):
			save_data = not save_data



if __name__=="__main__":
	main()


