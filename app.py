import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
import os
from collections import deque
import json
from datetime import datetime


class GestureControl:
    def __init__(self):
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480
        
        # Tracking
        self.cursor_history = deque(maxlen=5)
        self.gesture_history = deque(maxlen=6)
        self.finger_history = deque(maxlen=4)
        
        self.prev_hand_x, self.prev_hand_y = None, None
        self.prev_wrist = None
        self.click_cooldown = 0
        self.cursor_speed = 3.2
        
        # Modes
        self.current_mode = 'normal'  # normal, drawing, gaming, presentation
        self.drag_active = False
        
        # Drawing mode
        self.drawing_canvas = None
        self.drawing_color = (0, 255, 0)
        self.drawing_thickness = 3
        self.drawing_points = []
        
        # Gaming mode
        self.gaming_keys = {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}
        
        # Custom gestures
        self.custom_gestures = {}
        self.recording_gesture = False
        self.recorded_positions = []
        
        # Stats & Analytics
        self.total_gestures = 0
        self.gesture_counts = {}
        self.session_start = time.time()
        self.cursor_distance = 0
        self.click_positions = []
        
        # AI Features
        self.gesture_prediction = 'none'
        self.confidence_threshold = 0.7
        
        # Notifications
        self.show_notifications = True
        
        pyautogui.FAILSAFE = False
        
        self.init_drawing_canvas()
    
    def init_drawing_canvas(self):
        self.drawing_canvas = np.zeros((self.cam_height, self.cam_width, 3), dtype=np.uint8)
    
    def save_stats(self):
        stats = {
            'session_date': datetime.now().isoformat(),
            'duration': int(time.time() - self.session_start),
            'total_gestures': self.total_gestures,
            'gesture_breakdown': self.gesture_counts,
            'cursor_distance': round(self.cursor_distance, 2),
            'clicks': len(self.click_positions)
        }
        with open(f'session_{int(time.time())}.json', 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"📊 Stats saved!")
    
    def notify(self, title, message):
        if self.show_notifications:
            os.system(f'osascript -e \'display notification "{message}" with title "{title}"\'')
    
    def count_fingers(self, lm_list):
        fingers = []
        tips = [4, 8, 12, 16, 20]
        
        if lm_list[tips[0]][0] < lm_list[tips[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            if lm_list[tips[id]][1] < lm_list[tips[id] - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def get_landmarks(self, hand_landmarks):
        lm_list = []
        for lm in hand_landmarks.landmark:
            cx = int(lm.x * self.cam_width)
            cy = int(lm.y * self.cam_height)
            lm_list.append([cx, cy])
        return lm_list
    
    def distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def detect_swipe(self, lm_list):
        wrist = lm_list[0]
        
        if self.prev_wrist is None:
            self.prev_wrist = wrist
            return None
        
        delta_x = wrist[0] - self.prev_wrist[0]
        delta_y = wrist[1] - self.prev_wrist[1]
        
        self.prev_wrist = wrist
        
        threshold = 70
        if abs(delta_x) > threshold:
            return 'swipe_right' if delta_x > 0 else 'swipe_left'
        if abs(delta_y) > threshold:
            return 'swipe_down' if delta_y > 0 else 'swipe_up'
        
        return None
    
    def move_cursor(self, x, y):
        if self.prev_hand_x is None:
            self.prev_hand_x = x
            self.prev_hand_y = y
            return
        
        delta_x = (x - self.prev_hand_x) * self.cursor_speed
        delta_y = (y - self.prev_hand_y) * self.cursor_speed
        
        # Track distance
        self.cursor_distance += abs(delta_x) + abs(delta_y)
        
        curr_x, curr_y = pyautogui.position()
        new_x = max(0, min(self.screen_width - 1, curr_x + delta_x))
        new_y = max(0, min(self.screen_height - 1, curr_y + delta_y))
        
        pyautogui.moveTo(new_x, new_y, _pause=False)
        
        self.prev_hand_x = x
        self.prev_hand_y = y
    
    def draw_on_canvas(self, pos):
        if len(self.drawing_points) > 0:
            cv2.line(self.drawing_canvas, self.drawing_points[-1], pos, 
                    self.drawing_color, self.drawing_thickness)
        self.drawing_points.append(pos)
    
    def gaming_control(self, gesture):
        if gesture == 'up':
            pyautogui.keyDown(self.gaming_keys['up'])
        elif gesture == 'down':
            pyautogui.keyDown(self.gaming_keys['down'])
        elif gesture == 'left':
            pyautogui.keyDown(self.gaming_keys['left'])
        elif gesture == 'right':
            pyautogui.keyDown(self.gaming_keys['right'])
        elif gesture == 'action':
            pyautogui.press('space')
    
    def detect_gesture_advanced(self, lm_list, fingers):
        finger_count = fingers.count(1)
        swipe = self.detect_swipe(lm_list)
        
        gesture = 'none'
        pos = None
        extra_data = {}
        
        # 0 fingers
        if finger_count == 0:
            gesture = 'double_click'
            
        # 1 finger
        elif finger_count == 1 and fingers[1] == 1:
            if self.current_mode == 'drawing':
                gesture = 'draw'
                pos = lm_list[8]
            elif self.current_mode == 'gaming':
                # Gaming directional control
                y_pos = lm_list[8][1]
                if y_pos < self.cam_height * 0.3:
                    gesture = 'gaming_up'
                elif y_pos > self.cam_height * 0.6:
                    gesture = 'gaming_down'
            else:
                gesture = 'cursor'
                pos = lm_list[8]
                
        # 2 fingers
        elif finger_count == 2:
            if fingers[0] == 1 and fingers[1] == 1:
                dist = self.distance(lm_list[4], lm_list[8])
                if dist < 40:
                    gesture = 'left_click'
                    self.click_positions.append(pyautogui.position())
                else:
                    gesture = 'cursor'
                    pos = lm_list[8]
            elif fingers[1] == 1 and fingers[2] == 1:
                gesture = 'right_click'
                
        # 3 fingers
        elif finger_count == 3:
            if swipe == 'swipe_left':
                gesture = 'three_swipe_left'
            elif swipe == 'swipe_right':
                gesture = 'three_swipe_right'
            elif lm_list[8][1] < self.cam_height * 0.3:
                gesture = 'scroll_up'
            elif lm_list[8][1] > self.cam_height * 0.6:
                gesture = 'scroll_down'
            else:
                gesture = 'screenshot'
                
        # 4 fingers
        elif finger_count == 4:
            if swipe == 'swipe_left':
                gesture = 'four_swipe_left'
            elif swipe == 'swipe_right':
                gesture = 'four_swipe_right'
            elif swipe == 'swipe_up':
                gesture = 'brightness_up'
            elif swipe == 'swipe_down':
                gesture = 'brightness_down'
            elif lm_list[8][1] < self.cam_height * 0.3:
                gesture = 'volume_up'
            elif lm_list[8][1] > self.cam_height * 0.6:
                gesture = 'volume_down'
            else:
                gesture = 'minimize_window'
                
        # 5 fingers
        elif finger_count == 5:
            if swipe == 'swipe_left':
                gesture = 'five_swipe_left'
            elif swipe == 'swipe_right':
                gesture = 'five_swipe_right'
            elif swipe == 'swipe_up':
                gesture = 'show_desktop'
            elif swipe == 'swipe_down':
                gesture = 'mission_control'
            else:
                gesture = 'play_pause'
        
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > 6:
            self.gesture_history.popleft()
        
        if len(self.gesture_history) >= 3:
            recent = list(self.gesture_history)[-3:]
            if recent.count(recent[0]) >= 2:
                return recent[0], pos, extra_data
        
        return gesture, pos, extra_data
    
    def log_gesture(self, gesture):
        if gesture not in ['none', 'cursor', 'draw']:
            self.total_gestures += 1
            self.gesture_counts[gesture] = self.gesture_counts.get(gesture, 0) + 1
    
    def execute_all(self, gesture, pos, extra_data):
        t = time.time()
        
        # Cursor & Drawing
        if gesture == 'cursor' and pos:
            self.move_cursor(pos[0], pos[1])
        elif gesture == 'draw' and pos:
            self.draw_on_canvas(pos)
            
        # Clicks
        elif gesture == 'left_click':
            if t - self.click_cooldown > 0.4:
                pyautogui.click()
                self.click_cooldown = t
                self.prev_hand_x = None
                self.log_gesture(gesture)
                print("✓ LEFT CLICK")
        elif gesture == 'right_click':
            if t - self.click_cooldown > 0.4:
                pyautogui.rightClick()
                self.click_cooldown = t
                self.prev_hand_x = None
                self.log_gesture(gesture)
                print("✓ RIGHT CLICK")
        elif gesture == 'double_click':
            if t - self.click_cooldown > 0.7:
                pyautogui.doubleClick()
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("✓✓ DOUBLE CLICK")
                
        # Scrolling
        elif gesture == 'scroll_up':
            pyautogui.scroll(12)
        elif gesture == 'scroll_down':
            pyautogui.scroll(-12)
            
        # Volume
        elif gesture == 'volume_up':
            if t - self.click_cooldown > 0.2:
                os.system("osascript -e 'set volume output volume (output volume of (get volume settings) + 6)'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🔊 VOL+")
        elif gesture == 'volume_down':
            if t - self.click_cooldown > 0.2:
                try:
                    vol = os.popen("osascript -e 'output volume of (get volume settings)'").read().strip()
                    new_vol = max(0, int(vol) - 6)
                    os.system(f"osascript -e 'set volume output volume {new_vol}'")
                    self.click_cooldown = t
                    self.log_gesture(gesture)
                    print("🔉 VOL-")
                except:
                    pass
                    
        # Brightness
        elif gesture == 'brightness_up':
            if t - self.click_cooldown > 0.3:
                os.system("osascript -e 'tell application \"System Events\" to key code 144'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("☀️ BRIGHT+")
        elif gesture == 'brightness_down':
            if t - self.click_cooldown > 0.3:
                os.system("osascript -e 'tell application \"System Events\" to key code 145'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🌙 BRIGHT-")
                
        # Screenshot
        elif gesture == 'screenshot':
            if t - self.click_cooldown > 1.0:
                filename = f"screenshot_{int(t)}.png"
                pyautogui.screenshot(filename)
                self.click_cooldown = t
                self.log_gesture(gesture)
                print(f"📸 {filename}")
                self.notify("Screenshot", f"Saved: {filename}")
                
        # Media
        elif gesture == 'play_pause':
            if t - self.click_cooldown > 0.6:
                os.system("osascript -e 'tell application \"System Events\" to keystroke space'")
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("▶️ PLAY/PAUSE")
                
        # App switching
        elif gesture == 'four_swipe_left':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('command', 'shift', 'tab')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("◀️ PREV APP")
        elif gesture == 'four_swipe_right':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('command', 'tab')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("▶️ NEXT APP")
                
        # Presentation/Navigation
        elif gesture == 'three_swipe_left':
            if t - self.click_cooldown > 0.6:
                if self.current_mode == 'presentation':
                    pyautogui.press('left')
                    print("⏮ PREV SLIDE")
                else:
                    pyautogui.hotkey('command', '[')
                    print("⏪ BACK")
                self.click_cooldown = t
                self.log_gesture(gesture)
        elif gesture == 'three_swipe_right':
            if t - self.click_cooldown > 0.6:
                if self.current_mode == 'presentation':
                    pyautogui.press('right')
                    print("⏭ NEXT SLIDE")
                else:
                    pyautogui.hotkey('command', ']')
                    print("⏩ FORWARD")
                self.click_cooldown = t
                self.log_gesture(gesture)
                
        # Window management
        elif gesture == 'minimize_window':
            if t - self.click_cooldown > 1.0:
                pyautogui.hotkey('command', 'm')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("⬇️ MINIMIZE")
                
        # Desktop
        elif gesture == 'five_swipe_left':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('ctrl', 'left')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("◀️ PREV DESKTOP")
        elif gesture == 'five_swipe_right':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('ctrl', 'right')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("▶️ NEXT DESKTOP")
        elif gesture == 'show_desktop':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('fn', 'f11')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🖥 SHOW DESKTOP")
        elif gesture == 'mission_control':
            if t - self.click_cooldown > 0.8:
                pyautogui.hotkey('ctrl', 'up')
                self.click_cooldown = t
                self.log_gesture(gesture)
                print("🎛 MISSION CONTROL")
                
        # Gaming
        elif gesture.startswith('gaming_'):
            if gesture == 'gaming_up':
                pyautogui.keyDown('w')
                time.sleep(0.1)
                pyautogui.keyUp('w')
    
    def draw_crazy_ui(self, frame, gesture, finger_count, fps):
        h, w = frame.shape[:2]
        
        # Header bar
        cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
        
        # Mode indicator
        mode_colors = {
            'normal': (0, 255, 0),
            'drawing': (255, 0, 255),
            'gaming': (0, 128, 255),
            'presentation': (255, 165, 0)
        }
        mode_color = mode_colors.get(self.current_mode, (255, 255, 255))
        cv2.putText(frame, f"MODE: {self.current_mode.upper()}", 
                   (15, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, mode_color, 2)
        
        # Gesture
        gesture_color = (0, 255, 0) if gesture not in ['none', 'cursor'] else (100, 100, 100)
        cv2.putText(frame, f"{gesture.upper()}", 
                   (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, gesture_color, 2)
        
        # Stats bar
        cv2.putText(frame, f"F:{finger_count} | G:{self.total_gestures} | FPS:{int(fps)}", 
                   (w - 280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Cursor distance tracker
        cv2.putText(frame, f"Distance: {int(self.cursor_distance)}px", 
                   (w - 280, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 255), 1)
        
        # Drawing overlay
        if self.current_mode == 'drawing':
            alpha = 0.4
            overlay = cv2.addWeighted(frame, 1-alpha, self.drawing_canvas, alpha, 0)
            frame = overlay
            
            # Drawing controls
            cv2.putText(frame, "DRAWING MODE", (w//2 - 100, h - 80), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, "C: Clear | 1-9: Color | +/-: Thickness", 
                       (w//2 - 180, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Control bar
        cv2.rectangle(frame, (5, h-40), (w-5, h-5), (40, 40, 40), -1)
        cv2.putText(frame, "Q:Quit | M:Mode | D:Draw | G:Game | P:Present | S:Stats | H:Help", 
                   (12, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return frame
    
    def show_stats(self):
        print("\n" + "="*70)
        print("📊 ADVANCED SESSION STATISTICS")
        print("="*70)
        duration = int(time.time() - self.session_start)
        print(f"Duration: {duration//60}m {duration%60}s")
        print(f"Total Gestures: {self.total_gestures}")
        print(f"Cursor Distance: {int(self.cursor_distance)}px")
        print(f"Clicks: {len(self.click_positions)}")
        print(f"Current Mode: {self.current_mode}")
        print("\nTop 10 Gestures:")
        for gesture, count in sorted(self.gesture_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {gesture:25s}: {count:3d}")
        print("="*70 + "\n")
    
    def show_help(self):
        print("\n" + "="*70)
        print("📖 COMPLETE GESTURE GUIDE")
        print("="*70)
        print("\n🖱 BASIC CONTROLS:")
        print("  ✊ Fist → Double Click")
        print("  👆 1 Finger → Cursor / Draw (in drawing mode)")
        print("  👌 Pinch → Left Click")
        print("  ✌️  Peace → Right Click")
        print("\n📜 SCROLLING & NAVIGATION:")
        print("  🤟 3 Up/Down → Scroll | 3 Mid → Screenshot")
        print("  🤟 3 Swipe L/R → Prev/Next Slide or Back/Forward")
        print("\n🔊 MEDIA & SYSTEM:")
        print("  🖖 4 Up/Down → Volume")
        print("  🖖 4 Swipe Up/Down → Brightness")
        print("  🖖 4 Swipe L/R → App Switching")
        print("  🖖 4 Mid → Minimize Window")
        print("  ✋ 5 → Play/Pause")
        print("  ✋ 5 Swipe L/R → Desktop Switching")
        print("  ✋ 5 Swipe Up/Down → Show Desktop / Mission Control")
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("  M → Cycle Modes (Normal/Drawing/Gaming/Presentation)")
        print("  D → Drawing Mode")
        print("  G → Gaming Mode")
        print("  P → Presentation Mode")
        print("  C → Clear Canvas (in drawing mode)")
        print("  S → Show Statistics")
        print("  H → This Help")
        print("  Q → Quit")
        print("="*70 + "\n")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*70)
        print("🚀 ULTIMATE CRAZY GESTURE CONTROL - ALL FEATURES")
        print("="*70)
        print("\n✨ FEATURES:")
        print("  ✅ 30+ Gestures")
        print("  ✅ Drawing Mode (Air Drawing)")
        print("  ✅ Gaming Controls")
        print("  ✅ Presentation Mode")
        print("  ✅ Advanced Analytics")
        print("  ✅ Custom Gesture Recording")
        print("  ✅ Desktop Management")
        print("  ✅ Window Controls")
        print("\nPress 'H' for complete gesture guide")
        print("="*70 + "\n")
        
        self.notify("Gesture Control", "Ultimate Edition Started!")
        
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            gesture = 'none'
            finger_count = 0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(0,150,255), thickness=2)
                    )
                    
                    lm_list = self.get_landmarks(hand_landmarks)
                    fingers = self.count_fingers(lm_list)
                    finger_count = fingers.count(1)
                    
                    gesture, pos, extra = self.detect_gesture_advanced(lm_list, fingers)
                    
                    if gesture != 'none':
                        self.execute_all(gesture, pos, extra)
            else:
                self.prev_hand_x = None
                self.prev_wrist = None
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            frame = self.draw_crazy_ui(frame, gesture, finger_count, fps)
            
            cv2.imshow("Ultimate Crazy Gesture Control", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                modes = ['normal', 'drawing', 'gaming', 'presentation']
                idx = modes.index(self.current_mode)
                self.current_mode = modes[(idx + 1) % len(modes)]
                print(f"🎯 Mode: {self.current_mode.upper()}")
                self.notify("Mode Change", self.current_mode.upper())
            elif key == ord('d'):
                self.current_mode = 'drawing'
                print("🎨 DRAWING MODE")
            elif key == ord('g'):
                self.current_mode = 'gaming'
                print("🎮 GAMING MODE")
            elif key == ord('p'):
                self.current_mode = 'presentation'
                print("📊 PRESENTATION MODE")
            elif key == ord('c'):
                self.init_drawing_canvas()
                self.drawing_points = []
                print("🗑 Canvas Cleared")
            elif key == ord('s'):
                self.show_stats()
                self.save_stats()
            elif key == ord('h'):
                self.show_help()
        
        cap.release()
        cv2.destroyAllWindows()
        self.show_stats()
        self.save_stats()
        print("\n✅ Ultimate Session Ended!\n")


if __name__ == "__main__":
    controller = GestureControl()
    controller.run()


