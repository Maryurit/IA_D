# Instrucciones de Prueba - Streaming MJPEG de Cámaras

## Resumen de Implementación

Se ha implementado un servidor de streaming MJPEG que permite visualizar las cámaras procesadas por la IA en tiempo real desde el frontend.

## Archivos Modificados/Creados

1. **`streaming_server.py`** (NUEVO) - Servidor Flask para streaming MJPEG
2. **`detection_service.py`** (MODIFICADO) - Integra envío de frames al servidor
3. **`requirements.txt`** (MODIFICADO) - Agregado Flask como dependencia

## Pasos para Instalar y Ejecutar

### 1. Instalar Dependencias

```bash
cd depamanager-ia
pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

Asegúrate de que el archivo `.env` en `depamanager-ia` tenga las siguientes variables:

```env
BACKEND_URL=http://localhost:3000
SERVICE_TOKEN=tu_service_token_del_backend
CHECK_CAMERAS_INTERVAL=25
```

### 3. Iniciar el Backend

```bash
cd depamanager-backend
npm start
```

### 4. Crear una Cámara de Prueba en el Backend

Usa Postman o el frontend para crear una cámara:

```
POST http://localhost:3000/api/camaras
Authorization: Bearer {token_propietario}
Content-Type: application/json

{
  "nombre": "Cámara Test",
  "ubicacion": "Entrada Principal",
  "urlStream": "rtsp://localhost:8554/camera-test",
  "edificioId": "{edificio_id}"
}
```

**Nota**: Si no tienes una cámara RTSP real, puedes usar una cámara de prueba con un video:

```bash
# Usar FFmpeg para simular cámara RTSP desde un video
ffmpeg -re -i tu_video.mp4 -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/camera-test
```

O usar una cámara web como RTSP:

```bash
ffmpeg -f dshow -i video="Nombre de tu cámara" -c:v libx264 -preset ultrafast -f rtsp rtsp://localhost:8554/camera-webcam
```

### 5. Iniciar el Servicio de IA con Streaming

```bash
cd depamanager-ia
python detection_service.py
```

Deberías ver:
```
🚀 DepaManager LPR Multi-Cámara iniciado
📡 Servidor de streaming MJPEG iniciado en puerto 5001
```

### 6. Verificar que el Servidor de Streaming Funciona

Abre en tu navegador:

```
http://localhost:5001/health
```

Deberías ver:
```json
{
  "status": "ok",
  "active_cameras": 1,
  "camera_ids": ["uuid-de-tu-camara"]
}
```

### 7. Probar el Stream de la Cámara

Abre en tu navegador:

```
http://localhost:5001/stream/{camera_id}
```

Deberías ver el video de la cámara en tiempo real.

**Nota**: Reemplaza `{camera_id}` con el ID de tu cámara (puedes obtenerlo del backend).

### 8. Probar en el Frontend

#### Opción A: HTML Simple

Crea un archivo `test-camera.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Test Cámara</title>
</head>
<body>
    <h1>Cámara en Vivo</h1>
    <img src="http://localhost:5001/stream/{camera_id}" 
         alt="Cámara en vivo" 
         style="width: 640px; height: 360px;" />
</body>
</html>
```

#### Opción B: React Component

```jsx
import React, { useState, useEffect } from 'react';

const CameraLive = ({ cameraId, cameraName }) => {
  const streamUrl = `http://localhost:5001/stream/${cameraId}`;
  
  return (
    <div className="camera-container">
      <h3>{cameraName}</h3>
      <img 
        src={streamUrl} 
        alt={cameraName}
        style={{ width: '100%', height: 'auto', maxWidth: '640px' }}
        onError={(e) => {
          console.error('Error cargando stream:', e);
          e.target.src = 'https://via.placeholder.com/640x360?text=Cámara+no+disponible';
        }}
      />
    </div>
  );
};

export default CameraLive;
```

#### Opción C: Con Backend API

Primero, crea un endpoint en el backend para obtener cámaras activas:

```javascript
// En camaras.controller.js
async obtenerCamarasActivas(req, res) {
  try {
    const camaras = await camarasRepository.findByEdificios(req.user.edificiosIds);
    return success(res, camaras, 'Cámaras obtenidas correctamente');
  } catch (err) {
    return error(res, err.message, 500);
  }
}
```

Luego en el frontend:

```jsx
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const CamerasList = () => {
  const [cameras, setCameras] = useState([]);

  useEffect(() => {
    const fetchCameras = async () => {
      const response = await api.get('/camaras/activas');
      setCameras(response.data);
    };
    fetchCameras();
  }, []);

  return (
    <div>
      <h2>Cámaras en Vivo</h2>
      {cameras.map(camera => (
        <div key={camera.id}>
          <h3>{camera.nombre}</h3>
          <img 
            src={`http://localhost:5001/stream/${camera.id}`}
            alt={camera.nombre}
            style={{ width: '100%', maxWidth: '640px' }}
          />
        </div>
      ))}
    </div>
  );
};
```

## Solución de Problemas

### El servidor de streaming no inicia

**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solución**:
```bash
pip install flask
```

### Error: "Cámara no disponible" en el stream

**Causa**: La cámara no está siendo procesada por la IA.

**Solución**:
1. Verifica que la cámara esté activa en el backend
2. Verifica que el servicio de IA esté corriendo
3. Verifica que la URL RTSP sea correcta
4. Revisa los logs del servicio de IA

### Error: Conexión RTSP fallida

**Causa**: La cámara RTSP no está disponible.

**Solución**:
1. Verifica que la cámara esté encendida
2. Verifica la URL RTSP
3. Prueba la URL RTSP con VLC o FFmpeg primero

### El stream es lento o se corta

**Causa**: Latencia de red o procesamiento pesado.

**Solución**:
1. Reduce la resolución de los frames en `detection_service.py`
2. Aumenta `FRAME_SKIP` para procesar menos frames
3. Verifica que tu red tenga suficiente ancho de banda

## Arquitectura

```
┌─────────────┐
│  Cámara RTSP │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ detection_service.py│
│ (Procesa frames)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ streaming_server.py │
│ (Servidor MJPEG)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Frontend Web      │
│  (Muestra stream)  │
└─────────────────────┘
```

## Notas Importantes

1. **Puerto**: El servidor de streaming usa el puerto 5001. Asegúrate de que no esté en uso.
2. **Latencia**: La latencia típica es de 100-300ms dependiendo de la red.
3. **Ancho de banda**: Cada cámara consume ~1-2 Mbps de ancho de banda.
4. **Seguridad**: En producción, agrega autenticación al servidor de streaming.
5. **HTTPS**: En producción, usa HTTPS para el streaming.

## Próximos Pasos (Opcional)

Para producción, considera:

1. **Autenticación**: Agregar token de autenticación al servidor de streaming
2. **HTTPS**: Usar certificado SSL para streaming seguro
3. **Balanceo de carga**: Si tienes muchas cámaras, usa balanceo de carga
4. **Grabación**: Agregar funcionalidad para grabar streams
5. **WebRTC**: Para menor latencia, considera migrar a WebRTC
