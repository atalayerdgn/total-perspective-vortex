#!/bin/bash

# EEG/BCI Docker Volume Management Script

set -e

COMPOSE_FILE="docker-compose.yml"
DATA_VOLUME="eeg_data"
MODELS_VOLUME="eeg_models"

echo "=== EEG/BCI Docker Volume Yönetimi ==="

show_usage() {
    echo "Kullanım: $0 [komut]"
    echo ""
    echo "Komutlar:"
    echo "  start     - Container'ları başlat"
    echo "  stop      - Container'ları durdur"
    echo "  restart   - Container'ları yeniden başlat"
    echo "  logs      - Container loglarını göster"
    echo "  shell     - BCI app container'ına bağlan"
    echo "  jupyter   - Jupyter notebook'u başlat"
    echo "  volumes   - Volume bilgilerini göster"
    echo "  backup    - Volume'ları yedekle"
    echo "  restore   - Volume'ları geri yükle"
    echo "  clean     - Tüm container ve volume'ları temizle"
    echo "  help      - Bu yardım mesajını göster"
}

start_services() {
    echo "📦 Container'lar başlatılıyor..."
    docker-compose up -d bci-app
    echo "✅ BCI app başlatıldı!"
    echo "Shell erişimi için: docker-compose exec bci-app /bin/bash"
}

stop_services() {
    echo "🛑 Container'lar durduruluyor..."
    docker-compose down
    echo "✅ Container'lar durduruldu!"
}

restart_services() {
    echo "🔄 Container'lar yeniden başlatılıyor..."
    docker-compose down
    docker-compose up -d bci-app
    echo "✅ Container'lar yeniden başlatıldı!"
}

show_logs() {
    echo "📋 Container logları:"
    docker-compose logs -f bci-app
}

open_shell() {
    echo "🐚 BCI app shell açılıyor..."
    docker-compose exec bci-app /bin/bash
}

start_jupyter() {
    echo "📓 Jupyter notebook başlatılıyor..."
    docker-compose --profile dev up -d jupyter
    echo "✅ Jupyter başlatıldı!"
    echo "🌐 Tarayıcıda http://localhost:8888 adresini açın"
    echo "📋 Token için: docker-compose logs jupyter"
}

show_volumes() {
    echo "💾 Volume bilgileri:"
    echo ""
    echo "📊 Volume listesi:"
    docker volume ls | grep -E "(eeg_data|eeg_models)" || echo "❌ EEG volume'ları bulunamadı"
    echo ""
    echo "📈 Volume boyutları:"
    docker system df -v | grep -A 20 "Local Volumes:" | grep -E "(eeg_data|eeg_models)" || echo "❌ Volume boyut bilgisi alınamadı"
    echo ""
    echo "🔍 Volume detayları:"
    docker volume inspect $DATA_VOLUME 2>/dev/null || echo "❌ $DATA_VOLUME volume'u bulunamadı"
    docker volume inspect $MODELS_VOLUME 2>/dev/null || echo "❌ $MODELS_VOLUME volume'u bulunamadı"
}

backup_volumes() {
    BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    echo "💾 Volume'lar yedekleniyor: $BACKUP_DIR"
    
    # Data volume backup
    echo "📦 Data volume yedekleniyor..."
    docker run --rm -v $DATA_VOLUME:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine \
        tar czf /backup/eeg_data.tar.gz -C /data .
    
    # Models volume backup
    echo "🤖 Models volume yedekleniyor..."
    docker run --rm -v $MODELS_VOLUME:/models -v "$(pwd)/$BACKUP_DIR":/backup alpine \
        tar czf /backup/eeg_models.tar.gz -C /models .
    
    echo "✅ Yedekleme tamamlandı: $BACKUP_DIR"
    ls -lh "$BACKUP_DIR"
}

restore_volumes() {
    if [ -z "$1" ]; then
        echo "❌ Geri yükleme dizini belirtmelisiniz!"
        echo "Kullanım: $0 restore <backup_dizini>"
        echo "Mevcut yedekler:"
        ls -la ./backups/ 2>/dev/null || echo "Henüz yedek bulunamadı"
        exit 1
    fi
    
    BACKUP_DIR="$1"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "❌ Yedek dizini bulunamadı: $BACKUP_DIR"
        exit 1
    fi
    
    echo "📦 Volume'lar geri yükleniyor: $BACKUP_DIR"
    
    # Data volume restore
    if [ -f "$BACKUP_DIR/eeg_data.tar.gz" ]; then
        echo "📊 Data volume geri yükleniyor..."
        docker run --rm -v $DATA_VOLUME:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine \
            tar xzf /backup/eeg_data.tar.gz -C /data
    fi
    
    # Models volume restore
    if [ -f "$BACKUP_DIR/eeg_models.tar.gz" ]; then
        echo "🤖 Models volume geri yükleniyor..."
        docker run --rm -v $MODELS_VOLUME:/models -v "$(pwd)/$BACKUP_DIR":/backup alpine \
            tar xzf /backup/eeg_models.tar.gz -C /models
    fi
    
    echo "✅ Geri yükleme tamamlandı!"
}

clean_all() {
    echo "⚠️  UYARI: Bu işlem tüm container'ları ve volume'ları silecek!"
    read -p "Devam etmek istediğinizden emin misiniz? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🧹 Temizlik başlatılıyor..."
        docker-compose down -v
        docker volume rm $DATA_VOLUME $MODELS_VOLUME 2>/dev/null || true
        echo "✅ Temizlik tamamlandı!"
    else
        echo "❌ İşlem iptal edildi."
    fi
}

# Ana komut işleme
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs
        ;;
    shell)
        open_shell
        ;;
    jupyter)
        start_jupyter
        ;;
    volumes)
        show_volumes
        ;;
    backup)
        backup_volumes
        ;;
    restore)
        restore_volumes "$2"
        ;;
    clean)
        clean_all
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Bilinmeyen komut: $1"
        show_usage
        exit 1
        ;;
esac
