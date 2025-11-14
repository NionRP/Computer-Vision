import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import glob
import threading

# ====== МОДЕЛЬ НЕЙРОННОЙ СЕТИ ======
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ====== КАСТОМНЫЙ ДАТАСЕТ ======
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = []
        self.transform = transform
        
        for label in range(10):
            folder_path = os.path.join(data_dir, str(label))
            if os.path.exists(folder_path):
                for img_path in glob.glob(os.path.join(folder_path, '*.png')):
                    self.data.append((img_path, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ====== ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ ======
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание цифр - Neural Network")
        self.root.geometry("800x600")
        
        # Инициализация модели
        self.model = CNN()
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Переменные для путей
        self.model_path = tk.StringVar(value="model.pth")
        self.data_path = tk.StringVar()
        self.image_path = tk.StringVar()
        
        self.setup_ui()
        
        # Загружаем модель если существует
        self.load_model_if_exists()
    
    def setup_ui(self):
        # Создаем вкладки
        notebook = ttk.Notebook(self.root)
        
        # Вкладка распознавания
        self.recognition_frame = ttk.Frame(notebook)
        notebook.add(self.recognition_frame, text="Распознавание")
        
        # Вкладка обучения
        self.training_frame = ttk.Frame(notebook)
        notebook.add(self.training_frame, text="Обучение")
        
        notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        self.setup_recognition_tab()
        self.setup_training_tab()
    
    def setup_recognition_tab(self):
        # Заголовок
        ttk.Label(self.recognition_frame, text="Распознавание цифр", 
                 font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Фрейм для выбора модели
        model_frame = ttk.LabelFrame(self.recognition_frame, text="Модель", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(model_frame, text="Папка модели:").grid(row=0, column=0, sticky='w')
        ttk.Entry(model_frame, textvariable=self.model_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(model_frame, text="Обзор", 
                  command=self.browse_model_folder).grid(row=0, column=2, padx=5)
        
        # Фрейм для выбора изображения
        image_frame = ttk.LabelFrame(self.recognition_frame, text="Изображение", padding=10)
        image_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(image_frame, text="Файл изображения:").grid(row=0, column=0, sticky='w')
        ttk.Entry(image_frame, textvariable=self.image_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(image_frame, text="Обзор", 
                  command=self.browse_image).grid(row=0, column=2, padx=5)
        
        # Кнопка распознавания
        ttk.Button(self.recognition_frame, text="Распознать цифру", 
                  command=self.recognize_digit, style='Accent.TButton').pack(pady=10)
        
        # Область для отображения изображения
        self.image_label = ttk.Label(self.recognition_frame, text="Изображение появится здесь")
        self.image_label.pack(pady=10)
        
        # Область для результата
        self.result_label = ttk.Label(self.recognition_frame, text="", 
                                     font=('Arial', 14, 'bold'))
        self.result_label.pack(pady=10)
    
    def setup_training_tab(self):
        # Заголовок
        ttk.Label(self.training_frame, text="Обучение модели", 
                 font=('Arial', 16, 'bold')).pack(pady=10)
        
        # Информация о структуре папок
        info_frame = ttk.LabelFrame(self.training_frame, text="Требования к структуре данных", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = ("Папка должна содержать подпапки с названиями от 0 до 9.\n"
                    "В каждой подпапке должны находиться PNG-изображения соответствующей цифры.\n"
                    "Пример структуры:\n"
                    "data_folder/\n"
                    "├── 0/\n"
                    "│   ├── image1.png\n"
                    "│   └── image2.png\n"
                    "├── 1/\n"
                    "│   └── image3.png\n"
                    "...\n"
                    "└── 9/\n"
                    "    └── imageN.png")
        
        ttk.Label(info_frame, text=info_text, justify='left').pack(anchor='w')
        
        # Фрейм для данных обучения
        data_frame = ttk.LabelFrame(self.training_frame, text="Данные для обучения", padding=10)
        data_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(data_frame, text="Папка с данными:").grid(row=0, column=0, sticky='w')
        ttk.Entry(data_frame, textvariable=self.data_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(data_frame, text="Обзор", 
                  command=self.browse_data_folder).grid(row=0, column=2, padx=5)
        
        # Фрейм для параметров обучения
        params_frame = ttk.LabelFrame(self.training_frame, text="Параметры обучения", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(params_frame, text="Количество эпох:").grid(row=0, column=0, sticky='w')
        self.epochs_var = tk.StringVar(value="5")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(params_frame, text="Размер батча:").grid(row=0, column=2, sticky='w', padx=20)
        self.batch_size_var = tk.StringVar(value="64")
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3, sticky='w', padx=5)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(self.training_frame, mode='determinate')
        self.progress.pack(fill='x', padx=10, pady=10)
        
        # Текстовое поле для логов
        self.log_text = tk.Text(self.training_frame, height=10, width=80)
        self.log_text.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Кнопки обучения
        button_frame = ttk.Frame(self.training_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Начать обучение", 
                  command=self.start_training, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(button_frame, text="Остановить", 
                  command=self.stop_training).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Очистить логи", 
                  command=self.clear_logs).pack(side='left', padx=5)
        
        # Флаг для остановки обучения
        self.training_stopped = False
    
    def browse_model_folder(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
    
    def browse_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if path:
            self.image_path.set(path)
            self.display_image(path)
    
    def browse_data_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.data_path.set(path)
            # Проверяем структуру папки
            if not self.check_folder_structure(path):
                messagebox.showerror("Ошибка", "Неправильная структура папки!\n\n"
                                              "Папка должна содержать подпапки с названиями от 0 до 9.\n"
                                              "В каждой подпапке должны находиться PNG-изображения соответствующей цифры.")
                self.data_path.set("")  # Сбрасываем путь
    
    def check_folder_structure(self, folder_path):
        """Проверяет, соответствует ли структура папки требованиям"""
        # Проверяем наличие всех необходимых подпапок
        required_folders = [str(i) for i in range(10)]
        existing_folders = []
        
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item in required_folders:
                existing_folders.append(item)
                
                # Проверяем, есть ли в папке PNG-файлы
                png_files = glob.glob(os.path.join(item_path, "*.png"))
                if not png_files:
                    self.log(f"⚠ В папке {item} нет PNG-файлов")
                    return False
        
        # Проверяем, есть ли хотя бы одна папка с цифрами
        if not existing_folders:
            return False
            
        # Логируем информацию о найденных папках
        self.log(f"Найдены папки: {', '.join(sorted(existing_folders))}")
        missing_folders = set(required_folders) - set(existing_folders)
        if missing_folders:
            self.log(f"⚠ Отсутствуют папки: {', '.join(sorted(missing_folders))}")
            
        return True
    
    def display_image(self, path):
        try:
            image = Image.open(path)
            image.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
    def load_model_if_exists(self):
        if os.path.exists(self.model_path.get()):
            try:
                self.model.load_state_dict(torch.load(self.model_path.get(), map_location='cpu'))
                self.model.eval()
                self.log("✓ Модель загружена успешно")
            except Exception as e:
                self.log(f"✗ Ошибка загрузки модели: {e}")
    
    def recognize_digit(self):
        if not self.image_path.get():
            messagebox.showwarning("Предупреждение", "Выберите изображение для распознавания")
            return
        
        if not os.path.exists(self.model_path.get()):
            messagebox.showwarning("Предупреждение", "Модель не найдена. Сначала обучите модель.")
            return
        
        try:
            # Загружаем модель
            self.model.load_state_dict(torch.load(self.model_path.get(), map_location='cpu'))
            self.model.eval()
            
            # Загружаем и преобразуем изображение
            img = Image.open(self.image_path.get()).convert('L')
            tensor = self.transform(img).unsqueeze(0)
            
            # Распознаем
            with torch.no_grad():
                output = self.model(tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
            
            self.result_label.configure(text=f"Распознанная цифра: {prediction}")
            self.log(f"✓ Распознано: {prediction}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при распознавании: {e}")
    
    def start_training(self):
        if not self.data_path.get():
            messagebox.showwarning("Предупреждение", 
                                 "Выберите папку с данными для обучения")
            return
        
        # Проверяем структуру папки
        if not self.check_folder_structure(self.data_path.get()):
            messagebox.showerror("Ошибка", "Неправильная структура папки!\n\n"
                                          "Папка должна содержать подпапки с названиями от 0 до 9.\n"
                                          "В каждой подпапке должны находиться PNG-изображения соответствующей цифры.")
            return
        
        # Сбрасываем флаг остановки
        self.training_stopped = False
        
        # Запускаем обучение в отдельном потоке
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def stop_training(self):
        self.training_stopped = True
        self.log("⏹️ Обучение остановлено пользователем")
    
    def train_model(self):
        try:
            self.log("🎯 Начало обучения...")
            
            # Загружаем данные
            self.log("Загружаем кастомные данные...")
            trainset = CustomDataset(self.data_path.get(), transform=self.transform)
            
            if len(trainset) == 0:
                self.log("✗ В выбранной папке нет данных для обучения")
                return
                
            self.log(f"✓ Загружено {len(trainset)} изображений")
            
            batch_size = int(self.batch_size_var.get())
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            
            # Инициализация модели и оптимизатора
            model = CNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            epochs = int(self.epochs_var.get())
            
            # Обучение
            for epoch in range(epochs):
                if self.training_stopped:
                    break
                    
                running_loss = 0
                for i, (images, labels) in enumerate(trainloader):
                    if self.training_stopped:
                        break
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                    # Обновляем прогресс
                    progress = ((epoch * len(trainloader) + i) / (epochs * len(trainloader))) * 100
                    self.root.after(0, lambda: self.progress.configure(value=progress))
                
                avg_loss = running_loss / len(trainloader)
                self.log(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
            
            if not self.training_stopped:
                # Сохраняем модель
                torch.save(model.state_dict(), self.model_path.get())
                self.model = model
                self.log(f"✓ Модель сохранена в {self.model_path.get()}")
                self.log("🎉 Обучение завершено!")
                self.root.after(0, lambda: self.progress.configure(value=100))
                
        except Exception as e:
            self.log(f"✗ Ошибка при обучении: {e}")
    
    def log(self, message):
        def update_log():
            self.log_text.insert('end', f"{message}\n")
            self.log_text.see('end')
            self.root.update()
        
        self.root.after(0, update_log)
    
    def clear_logs(self):
        self.log_text.delete('1.0', 'end')

# ====== ЗАПУСК ПРИЛОЖЕНИЯ ======
if __name__ == "__main__":
    # Проверяем доступность библиотек
    try:
        root = tk.Tk()
        
        # Настраиваем стиль для акцентных кнопок
        style = ttk.Style()
        style.configure('Accent.TButton', foreground='Black', background='#0078d4')
        
        app = DigitRecognizerApp(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("Убедитесь, что установлены все необходимые библиотеки:")
        print("pip install torch torchvision pillow")