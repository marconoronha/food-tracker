import torch
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import os

def main ():
    # Verificar se a GPU está disponível e definir o dispositivo
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Definir transformações e carregar o dataset a partir de um diretório local
    data_dir = 'data'  # Substitua pelo caminho do seu diretório local

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Carregar modelo pré-treinado ResNet-18 e modificar a última camada
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Ajustar para o número de classes do dataset

    # Mover o modelo para o dispositivo (GPU)
    model = model.to(device)

    # Definir função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Configurar TensorBoard
    writer = SummaryWriter('runs/food101_resnet18_experiment')

    # Função de treinamento com checkpoints e TensorBoard
    def train_model(model, train_loader, criterion, optimizer, num_epochs=10, checkpoint_path='checkpoint.pth'):
        start_epoch = 0

        # Carregar checkpoint se disponível
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint loaded. Starting from epoch {start_epoch}")

        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0

            start_time = time.time()
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Mover dados para a GPU
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)
            epoch_time = time.time() - start_time

            print(f'Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time:.2f}s')
            
            # Escrever no TensorBoard
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', epoch_acc, epoch)

            # Salvar checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)

        writer.close()
        return model

    # Treinar o modelo
    model_trained = train_model(model, train_loader, criterion, optimizer, num_epochs=25)

    # Função de avaliação
    def evaluate_model(model, test_loader):
        model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  # Mover dados para a GPU
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the model on test images: %d %%' % (100 * correct / total))

    # Avaliar o modelo
    evaluate_model(model_trained, test_loader)

    # Salvar o modelo treinado
    torch.save(model_trained.state_dict(), 'food101_resnet18_model.pth')

if __name__ == '__main__':
    main()