import cv2
import numpy as np
import os
import joblib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
from cvzone.HandTrackingModule import HandDetector
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV



class MultiHandGestureSVM:
    def __init__(self):
        # Cấu hình HandDetector từ cvzone
        self.detector = HandDetector(detectionCon=0.5, maxHands=2)

        # Danh sách nhãn cho các cử chỉ
        self.gesture_labels = ['one', 'two', 'three', 'four', 'five']

        # SVM Classifier cho bàn tay trái và phải
        self.svm_classifier_left = SVC(kernel='rbf', probability=True, cache_size=2000)
        self.svm_classifier_right = SVC(kernel='rbf', probability=True, cache_size=2000)

        self.scaler_left = StandardScaler()
        self.scaler_right = StandardScaler()

        # Thêm tính tăng lưu trữ đệm
        self._feature_cache = {}

        # Khởi tạo nhóm luồng để xử lý song song
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())


    def extract_advanced_features(self, landmarks):
        """Trích xuất đặc trưng phong phú hơn từ landmarks"""
        # Kiểm tra bộ đệm
        landmark_key = tuple(landmarks)
        if landmark_key in self._feature_cache:
            return self._feature_cache[landmark_key]

        #  Chuyển đổi sang mảng có nhiều mảng để tính toán nhanh hơn
        landmarks_array = np.array(landmarks)
        features = []

        # Khoảng cách giữa các điểm
        coords = landmarks_array.reshape(-1, 2)
        distances = np.sqrt(np.sum((coords[:, np.newaxis] - coords) ** 2, axis=2))
        features.extend(distances[np.triu_indices(len(coords), k=1)])

        # Thêm các đặc trưng thống kê
        stats = np.array([
            np.mean(landmarks_array),
            np.std(landmarks_array),
            np.max(landmarks_array),
            np.min(landmarks_array)
        ])
        features.extend(stats)

        # Kết quả bộ nhớ đệm
        self._feature_cache[landmark_key] = features
        return features


    def _calculate_angle(self, p1, p2, p3):
        """Tính toán góc giữa 3 điểm"""
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)

    def extract_hand_landmarks(self, image):
        """Trích xuất landmarks cho cả hai bàn tay"""
        hands, img = self.detector.findHands(image, draw=False, flipType=True)

        hand_data = {'left': None, 'right': None}

        if hands:
            for hand in hands:
                handType = hand['type']
                lmList = hand['lmList']

                landmarks = []
                for lm in lmList:
                    landmarks.extend([lm[0], lm[1]])

                hand_data[handType.lower()] = landmarks

        return hand_data

    def process_image_batch(self, image_files, label, label_dir):
        """Xử lý hàng loạt hình ảnh song song"""
        results = []
        for image_file in image_files:
            image_path = os.path.join(label_dir, image_file)
            image = cv2.imread(image_path)
            if image is not None:
                hand_data = self.extract_hand_landmarks(image)
                if hand_data['left']:
                    left_features = self.extract_advanced_features(hand_data['left'])
                    results.append(('left', left_features, label))
                if hand_data['right']:
                    right_features = self.extract_advanced_features(hand_data['right'])
                    results.append(('right', right_features, label))
        return results

    def collect_training_data(self, data_dir='hand_gesture_data', batch_size=32):
        """Thu thập dữ liệu huấn luyện từ thư mục"""
        X_left, y_left, X_right, y_right = [], [], [], []


        for label in self.gesture_labels:
            label_dir = os.path.join(data_dir, label)
            if not os.path.exists(label_dir):
                print(f"Cảnh báo: Không tìm thấy thư mục {label_dir}")
                continue

            # Lấy tất cả các tập tin hình ảnh
            image_files = os.listdir(label_dir)

            # Xử lý hình ảnh theo từng đợt
            batches = [image_files[i:i + batch_size]
                       for i in range(0, len(image_files), batch_size)]

            # Xử lý các lô song song
            with ProcessPoolExecutor() as executor:
                batch_results = list(executor.map(
                    partial(self.process_image_batch, label=label, label_dir=label_dir),
                    batches
                ))

            # Kết hợp kết quả
            for batch in batch_results:
                for hand_type, features, label in batch:
                    if hand_type == 'left':
                        X_left.append(features)
                        y_left.append(label)
                    else:
                        X_right.append(features)
                        y_right.append(label)

            return (np.array(X_left), np.array(y_left)), \
                (np.array(X_right), np.array(y_right))

    def advanced_data_augmentation(self, X, y):
        """Kỹ thuật augmentation nâng cao cho SVM"""
        augmented_X, augmented_y = [], []

        for x, label in zip(X, y):
            augmented_X.append(x)
            augmented_y.append(label)

            # Thêm nhiễu Gaussian
            noise_level = 0.1
            noisy_x = x + np.random.normal(0, noise_level, x.shape)
            augmented_X.append(noisy_x)
            augmented_y.append(label)

            # Phép biến đổi tuyến tính
            scale_factor = np.random.uniform(0.9, 1.1)
            scaled_x = x * scale_factor
            augmented_X.append(scaled_x)
            augmented_y.append(label)

            # Kết hợp các phép biến đổi
        combined_x = scaled_x + np.random.normal(0, noise_level / 2, x.shape)
        augmented_X.append(combined_x)
        augmented_y.append(label)

        return np.array(augmented_X), np.array(augmented_y)

    def optimize_svm_parameters(self, X, y):
        """Tìm kiếm siêu tham số tối ưu"""
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }

        grid_search = GridSearchCV(
            SVC(probability=True, cache_size=2000),
            param_grid,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)
        print("Best parameters:", grid_search.best_params_)

        return grid_search.best_estimator_

    "Quá trình huấn luyện"
    def robust_train_svm(self, data):
        """Huấn luyện SVM với cross-validation"""
        (X_left, y_left), (X_right, y_right) = data

        # Kiểm tra dữ liệu
        if len(X_left) == 0 or len(X_right) == 0:
            print("Lỗi: Không có đủ dữ liệu để huấn luyện")
            return

        # Augmentation dữ liệu
        X_left_aug, y_left_aug = self.advanced_data_augmentation(X_left, y_left)
        X_right_aug, y_right_aug = self.advanced_data_augmentation(X_right, y_right)

        # Tối ưu và huấn luyện SVM cho bàn tay trái
        X_left_scaled = self.scaler_left.fit_transform(X_left_aug)
        self.svm_classifier_left = self.optimize_svm_parameters(X_left_scaled, y_left_aug)

        # Tối ưu và huấn luyện SVM cho bàn tay phải
        X_right_scaled = self.scaler_right.fit_transform(X_right_aug)
        self.svm_classifier_right = self.optimize_svm_parameters(X_right_scaled, y_right_aug)

        # Đánh giá với cross-validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        def evaluate_model(X, y, classifier, scaler):
            accuracies = []
            for train_idx, test_idx in kfold.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                classifier.fit(X_train_scaled, y_train)
                accuracy = classifier.score(X_test_scaled, y_test)
                accuracies.append(accuracy)

            return np.mean(accuracies)

        # Đánh giá song song
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_left = executor.submit(evaluate_model, X_left_aug, y_left_aug,
                                          self.svm_classifier_left, self.scaler_left)
            future_right = executor.submit(evaluate_model, X_right_aug, y_right_aug,
                                           self.svm_classifier_right, self.scaler_right)

            left_accuracy = future_left.result()
            right_accuracy = future_right.result()

        print(f"Độ chính xác của tay trái: {left_accuracy * 100:.2f}%")
        print(f"Độ chính xác của tay phải: {right_accuracy * 100:.2f}%")

        # Lưu mô hình
        joblib.dump(self.svm_classifier_left, 'hand_gesture_svm_left.joblib')
        joblib.dump(self.svm_classifier_right, 'hand_gesture_svm_right.joblib')
        joblib.dump(self.scaler_left, 'hand_gesture_scaler_left.joblib')
        joblib.dump(self.scaler_right, 'hand_gesture_scaler_right.joblib')

    def predict_gestures(self, image):
        """Dự đoán cử chỉ cho cả hai bàn tay"""
        hand_data = self.extract_hand_landmarks(image)
        predictions = {'left': 'Not detected', 'right': 'Not detected'}

        def predict_hand(hand_type):
            if hand_data[hand_type] and len(hand_data[hand_type]) > 0:
                features = self.extract_advanced_features(hand_data[hand_type])
                scaled = (self.scaler_left if hand_type == 'left' else self.scaler_right).transform([features])
                classifier = self.svm_classifier_left if hand_type == 'left' else self.svm_classifier_right
                return hand_type, classifier.predict(scaled)[0]
            return hand_type, 'Not detected'

        # Dự đoán song song cho cả hai tay
        future_left = self.executor.submit(predict_hand, 'left')
        future_right = self.executor.submit(predict_hand, 'right')

        left_type, left_pred = future_left.result()
        right_type, right_pred = future_right.result()

        predictions[left_type] = left_pred
        predictions[right_type] = right_pred

        return predictions

    def real_time_recognition(self):
        """Nhận dạng cử chỉ thời gian thực"""
        cap = cv2.VideoCapture(0)

        # Tối ưu hóa cài đặt camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Phân bổ trước bộ nhớ cho các khung
        frame_buffer = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sao chép khung vào bộ đệm
                np.copyto(frame_buffer, frame)

                # Dự đoán cử chỉ song song
                gestures = self.predict_gestures(frame_buffer)

                # Tối ưu hóa hình ảnh trực quan
                hands, img = self.detector.findHands(frame_buffer, draw=True, flipType=True)

                # Sử dụng kết xuất văn bản được tối ưu hóa
                cv2.putText(frame_buffer, f"Left: {gestures['left']}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_buffer, f"Right: {gestures['right']}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('Hand Gesture Recognition', frame_buffer)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.executor.shutdown()


# Sử dụng
def main():
    try:
        # Khởi tạo hệ thống
        hand_gesture_svm = MultiHandGestureSVM()

        # Kiểm tra xem các mô hình được đào tạo có tồn tại không
        if (os.path.exists('hand_gesture_svm_left.joblib') and
                os.path.exists('hand_gesture_svm_right.joblib')):
            print("Loading existing models...")
            hand_gesture_svm.svm_classifier_left = joblib.load('hand_gesture_svm_left.joblib')
            hand_gesture_svm.svm_classifier_right = joblib.load('hand_gesture_svm_right.joblib')
            hand_gesture_svm.scaler_left = joblib.load('hand_gesture_scaler_left.joblib')
            hand_gesture_svm.scaler_right = joblib.load('hand_gesture_scaler_right.joblib')
        else:
            print("Training new models...")
            training_data = hand_gesture_svm.collect_training_data()
            hand_gesture_svm.robust_train_svm(training_data)

        print("Starting real-time recognition...")
        hand_gesture_svm.real_time_recognition()

    except Exception as e:
        print(f"An error occurred:")


if __name__ == '__main__':
    main()