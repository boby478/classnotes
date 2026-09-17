package com.classnotes.capture;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioPlaybackCaptureConfiguration;
import android.media.AudioRecord;
import android.media.projection.MediaProjection;
import android.os.IBinder;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CaptureService extends Service {

    private static final String CHANNEL =
        "classnotes_capture";

    private static final String TRANSCRIBE_URL =
        "https://classnotes-i4nn.onrender.com/transcribe";

    private static final int SAMPLE_RATE = 44100;
    private static final int CHANNELS = 2;
    private static final int BYTES_PER_SAMPLE = 2;

    // 20 seconds per upload.
    private static final int CHUNK_SECONDS = 20;

    private MediaProjection projection;
    private AudioRecord recorder;

    private Thread captureThread;

    private volatile boolean running = false;

    private final ExecutorService uploadExecutor =
        Executors.newSingleThreadExecutor();

    @Override
    public void onCreate() {
        super.onCreate();

        NotificationChannel channel =
            new NotificationChannel(
                CHANNEL,
                "ClassNotes Capture",
                NotificationManager.IMPORTANCE_LOW
            );

        NotificationManager manager =
            (NotificationManager)
                getSystemService(
                    Context.NOTIFICATION_SERVICE
                );

        manager.createNotificationChannel(channel);
    }

    @Override
    public int onStartCommand(
        Intent intent,
        int flags,
        int startId
    ) {

        Notification notification =
            new Notification.Builder(
                this,
                CHANNEL
            )
            .setContentTitle(
                "ClassNotes is listening"
            )
            .setContentText(
                "Class audio capture is running"
            )
            .setSmallIcon(
                android.R.drawable.ic_btn_speak_now
            )
            .setOngoing(true)
            .build();

        startForeground(
            1,
            notification
        );

        int resultCode =
            intent.getIntExtra(
                "resultCode",
                -1
            );

        Intent data =
            intent.getParcelableExtra("data");

        android.media.projection.MediaProjectionManager manager =
            (android.media.projection.MediaProjectionManager)
                getSystemService(
                    MEDIA_PROJECTION_SERVICE
                );

        projection =
            manager.getMediaProjection(
                resultCode,
                data
            );

        if(projection == null){
            stopSelf();
            return START_NOT_STICKY;
        }

        startAudioCapture();

        return START_STICKY;
    }

    private void startAudioCapture() {

        AudioPlaybackCaptureConfiguration config =
            new AudioPlaybackCaptureConfiguration.Builder(
                projection
            )
            .addMatchingUsage(
                AudioAttributes.USAGE_MEDIA
            )
            .addMatchingUsage(
                AudioAttributes.USAGE_GAME
            )
            .addMatchingUsage(
                AudioAttributes.USAGE_UNKNOWN
            )
            .build();

        AudioFormat format =
            new AudioFormat.Builder()
                .setEncoding(
                    AudioFormat.ENCODING_PCM_16BIT
                )
                .setSampleRate(
                    SAMPLE_RATE
                )
                .setChannelMask(
                    AudioFormat.CHANNEL_IN_STEREO
                )
                .build();

        int buffer =
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_STEREO,
                AudioFormat.ENCODING_PCM_16BIT
            );

        recorder =
            new AudioRecord.Builder()
                .setAudioPlaybackCaptureConfig(
                    config
                )
                .setAudioFormat(format)
                .setBufferSizeInBytes(
                    Math.max(
                        buffer * 2,
                        65536
                    )
                )
                .build();

        recorder.startRecording();

        running = true;

        captureThread =
            new Thread(() -> {

                ByteArrayOutputStream chunk =
                    new ByteArrayOutputStream();

                int targetBytes =
                    SAMPLE_RATE *
                    CHANNELS *
                    BYTES_PER_SAMPLE *
                    CHUNK_SECONDS;

                short[] samples =
                    new short[8192];

                while(running){

                    int count =
                        recorder.read(
                            samples,
                            0,
                            samples.length
                        );

                    if(count <= 0)
                        continue;

                    for(int i = 0; i < count; i++){

                        short value =
                            samples[i];

                        chunk.write(
                            value & 0xff
                        );

                        chunk.write(
                            (value >> 8) & 0xff
                        );
                    }

                    if(chunk.size() >= targetBytes){

                        byte[] pcm =
                            chunk.toByteArray();

                        chunk.reset();

                        uploadExecutor.execute(
                            () -> uploadWav(pcm)
                        );
                    }
                }

                if(chunk.size() > 0){

                    byte[] pcm =
                        chunk.toByteArray();

                    uploadExecutor.execute(
                        () -> uploadWav(pcm)
                    );
                }

            });

        captureThread.start();
    }

    private byte[] makeWav(byte[] pcm){

        int dataLength = pcm.length;
        int totalLength = 36 + dataLength;

        ByteArrayOutputStream out =
            new ByteArrayOutputStream();

        try{

            DataOutputStream d =
                new DataOutputStream(out);

            d.writeBytes("RIFF");
            writeLEInt(d, totalLength);
            d.writeBytes("WAVE");

            d.writeBytes("fmt ");
            writeLEInt(d, 16);
            writeLEShort(d, (short)1);
            writeLEShort(d, (short)CHANNELS);
            writeLEInt(d, SAMPLE_RATE);

            int byteRate =
                SAMPLE_RATE *
                CHANNELS *
                BYTES_PER_SAMPLE;

            writeLEInt(d, byteRate);
            writeLEShort(
                d,
                (short)(
                    CHANNELS *
                    BYTES_PER_SAMPLE
                )
            );

            writeLEShort(
                d,
                (short)16
            );

            d.writeBytes("data");
            writeLEInt(d, dataLength);

            d.write(pcm);
            d.flush();

            return out.toByteArray();

        }catch(Exception e){

            return null;
        }
    }

    private void writeLEInt(
        DataOutputStream d,
        int value
    ) throws Exception {

        d.writeByte(value & 0xff);
        d.writeByte((value >> 8) & 0xff);
        d.writeByte((value >> 16) & 0xff);
        d.writeByte((value >> 24) & 0xff);
    }

    private void writeLEShort(
        DataOutputStream d,
        short value
    ) throws Exception {

        d.writeByte(value & 0xff);
        d.writeByte((value >> 8) & 0xff);
    }

    private void uploadWav(byte[] pcm){

        try{

            byte[] wav =
                makeWav(pcm);

            if(wav == null)
                return;

            String boundary =
                "----ClassNotes" +
                System.currentTimeMillis();

            URL url =
                new URL(TRANSCRIBE_URL);

            HttpURLConnection connection =
                (HttpURLConnection)
                    url.openConnection();

            connection.setRequestMethod("POST");
            connection.setDoOutput(true);

            connection.setConnectTimeout(30000);
            connection.setReadTimeout(300000);

            connection.setRequestProperty(
                "Content-Type",
                "multipart/form-data; boundary=" +
                boundary
            );

            DataOutputStream out =
                new DataOutputStream(
                    connection.getOutputStream()
                );

            out.writeBytes(
                "--" + boundary + "\r\n"
            );

            out.writeBytes(
                "Content-Disposition: form-data; " +
                "name=\"audio\"; " +
                "filename=\"class-chunk.wav\"\r\n"
            );

            out.writeBytes(
                "Content-Type: audio/wav\r\n\r\n"
            );

            out.write(wav);

            out.writeBytes(
                "\r\n--" +
                boundary +
                "--\r\n"
            );

            out.flush();
            out.close();

            int status =
                connection.getResponseCode();

            InputStream stream =
                status >= 400
                    ? connection.getErrorStream()
                    : connection.getInputStream();

            if(stream != null){

                byte[] buffer =
                    new byte[4096];

                while(
                    stream.read(buffer) != -1
                ){}
                
                stream.close();
            }

            connection.disconnect();

        }catch(Exception e){

            e.printStackTrace();
        }
    }

    @Override
    public void onDestroy(){

        running = false;

        if(recorder != null){

            try{
                recorder.stop();
            }catch(Exception ignored){}

            recorder.release();
            recorder = null;
        }

        if(projection != null){

            projection.stop();
            projection = null;
        }

        uploadExecutor.shutdown();

        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent){
        return null;
    }
}
