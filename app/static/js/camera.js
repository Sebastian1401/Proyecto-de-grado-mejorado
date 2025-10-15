
$(document).ready(function () {
    const cedula = window.__APP__.cedula;
    const captureUrlTpl = window.__APP__.capturaUrlTpl;

    // --- LÓGICA DE LA GALERÍA ---
    const galleryPanel = $('#galleryPanel');
    const galleryImages = $('#galleryImages');

    // --- LOGICA DE LAS PERILLAS ---
    const $body = $('body');
    const $knobsPanel = $('#knobsPanel');

    // Función para añadir una miniatura a la galería
    function addThumbnail(filename) {
        const thumbnailUrl = captureUrlTpl
            .replace('__CED__', cedula)
            .replace('__FILE__', filename) + `?t=${Date.now()}`;
        const item = $(`
                <div class="thumbnail-item" data-filename="${filename}">
                    <img src="${thumbnailUrl}" alt="Captura">
                    <button class="delete-btn"><i class="fas fa-times"></i></button>
                </div>
            `);
        galleryImages.append(item);
    }

    // Función para cargar todas las capturas existentes al iniciar
    function loadCaptures() {
        galleryImages.html('<div class="gallery-loader"><i class="fas fa-spinner fa-spin"></i> Cargando...</div>');

        $.getJSON(window.__APP__.getCapturas, function (files) {

            if (files.length === 0) {
                galleryImages.html('<p style="text-align:center; color:#888;">No hay capturas.</p>');
                return;
            }

            const tempContainer = $('<div></div>');
            let imagesLoaded = 0;

            files.forEach(filename => {
                const thumbnailUrl = captureUrlTpl
                    .replace('__CED__', cedula)
                    .replace('__FILE__', filename) + `?t=${Date.now()}`;
                const item = $(`
                        <div class="thumbnail-item" data-filename="${filename}">
                            <img src="${thumbnailUrl}" alt="Captura">
                            <button class="delete-btn"><i class="fas fa-times"></i></button>
                        </div>
                    `);

                item.find('img').on('load', function () {
                    imagesLoaded++;
                    if (imagesLoaded === files.length) {
                        galleryImages.html(tempContainer.html());
                    }
                });

                tempContainer.append(item);
            });
        });
    }

    // Evento para mostrar/ocultar la galería
    $('#galleryButton').click(function () {
        $('#galleryPanel').toggleClass('visible');
    });

    // Evento para borrar una imagen
    galleryImages.on('click', '.delete-btn', function () {
        const item = $(this).closest('.thumbnail-item');
        const filename = item.data('filename');
        const button = $(this);
        const originalIconClass = button.find('i').attr('class');

        setButtonLoading(button, true);

        $.ajax({
            url: '/delete_captura',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ filename: filename, cedula: cedula }),
            success: function (response) {
                if (response.success) {
                    showNotification('Imagen eliminada', 'success');
                    loadCaptures();
                } else {
                    showNotification(response.message || 'Error al eliminar', 'error');
                }
            },
            error: function () {
                showNotification('Error de comunicación al eliminar', 'error');
            },
            complete: function () {
                setButtonLoading(button, false);
                button.find('i').removeClass().addClass(originalIconClass);
            }
        });
    });

    // --- LÓGICA DE CAPTURA ---
    $('#captureButton').click(function () {
        const video = document.getElementById('video-frame');
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        var button = $(this);
        var originalIcon = button.find('i').attr('class');

        setButtonLoading(button, true);

        canvas.width = video.naturalWidth;
        canvas.height = video.naturalHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function (blob) {
            const formData = new FormData();
            formData.append('cedula', cedula);
            formData.append('image', blob, 'captura.jpg');

            $.ajax({
                url: window.__APP__.captureUrl,
                type: 'POST',
                data: formData,
                processData: false,
                contentType: false,
                success: function (data) {
                    showNotification(data.message || 'Imagen capturada', 'success');
                    if (data.filename) {
                        loadCaptures();
                    }
                },
                error: function () {
                    showNotification('Error al capturar', 'error');
                },
                complete: function () {
                    setButtonLoading(button, false);
                    button.find('i').removeClass().addClass(originalIcon);
                }
            });
        }, 'image/jpeg');
    });

    // --- OTRAS FUNCIONES ---
    $('#backButton').click(() => { window.location.href = window.__APP__.backUrl; });

    $('#predictionSwitch').change(function () {
        $.post(window.__APP__.togglePred, { enabled: this.checked });
    });

    // --- perillas RKNN ---
    function setKnobTexts(t) {
        $('#conf_th_val').text(Number(t.conf_th).toFixed(2));
        $('#iou_th_val').text(Number(t.iou_th).toFixed(2));
        $('#min_box_frac_val').text(Number(t.min_box_frac).toFixed(3));
    }

    function readKnobs() {
        return {
            conf_th: parseFloat($('#conf_th').val()),
            iou_th: parseFloat($('#iou_th').val()),
            min_box_frac: parseFloat($('#min_box_frac').val())
        };
    }

    let postTimer = null;
    $('#conf_th, #iou_th, #min_box_frac').on('input change', function () {
        const t = readKnobs();
        setKnobTexts(t);
        clearTimeout(postTimer);
        postTimer = setTimeout(() => {
            $.ajax({
                url: window.__APP__.thresholdsPost,
                type: 'POST',
                data: JSON.stringify(t),
                contentType: 'application/json'
            });
        }, 120);
    });

    function applyThresholds(t) {
        $('#conf_th').val(t.conf_th);
        $('#iou_th').val(t.iou_th);
        $('#min_box_frac').val(t.min_box_frac);
        setKnobTexts(t);
    }

    function loadThresholds() {
        return $.getJSON(window.__APP__.thresholdsGet).done(applyThresholds);
    }

    // botón restaurar
    $('#resetKnobs').on('click', function () {
        const $btn = $(this);
        const $icon = $btn.find('i');
        const orig = $icon.attr('class');

        setButtonLoading($btn, true);
        $.post(window.__APP__.thresholdsReset)
            .done(applyThresholds)
            .fail(() => showNotification('No se pudo restaurar perillas', 'error'))
            .always(() => {
                setButtonLoading($btn, false);
                $icon.attr('class', orig);   // <- restaura el icono
            });
    });


    // cargar al entrar a la página
    loadThresholds();

    function showNotification(message, type = 'success') {
        const notification = $(`
                <div class="notification ${type}" style="
                    position: fixed; top: 20px; right: 20px;
                    background: ${type === 'success' ? 'linear-gradient(135deg, #27ae60, #2ecc71)' : 'linear-gradient(135deg, #e74c3c, #c0392b)'};
                    color: white; padding: 15px 25px; border-radius: 12px;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.2); z-index: 1000;
                    font-family: 'Poppins', sans-serif; font-weight: 500;
                    display: flex; align-items: center; gap: 10px;
                    transform: translateX(400px); transition: all 0.4s ease;
                ">
                    <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
                    ${message}
                </div>
            `);
        $('body').append(notification);
        setTimeout(() => { notification.css('transform', 'translateX(0)'); }, 100);
        setTimeout(() => {
            notification.css('transform', 'translateX(400px)');
            setTimeout(() => notification.remove(), 400);
        }, 3000);
    }

    function setButtonLoading(button, loading = true) {
        if (loading) {
            button.addClass('loading');
            button.find('i').removeClass().addClass('fas fa-spinner fa-spin');
        } else {
            button.removeClass('loading');
        }
    }

    function setKnobsVisible(visible) {
        $body.toggleClass('knobs-visible', !!visible);
        $knobsPanel.attr('aria-hidden', (!visible).toString());
    }

    // Atajo: Shift+K alterna; Esc cierra
    document.addEventListener('keydown', (e) => {
        // Evitar si se escribe en inputs/select/textarea
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        const typing = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;

        // Shift+K (o Shift+k)
        if (e.shiftKey && (e.key === 'K' || e.key === 'k')) {
            if (!typing) {
                setKnobsVisible(!$body.hasClass('knobs-visible'));
                e.preventDefault();
            }
        }

        // Esc cierra
        if (e.key === 'Escape' && $body.hasClass('knobs-visible')) {
            setKnobsVisible(false);
        }
    });

    // Inicia oculto
    setKnobsVisible(false);

    loadCaptures();
});
