package com.classnotes.capture;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

public class MainActivity extends Activity {

    private static final int REQUEST_CAPTURE = 1001;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        TextView title = new TextView(this);
        title.setText(
            "ClassNotes Capture\n\n" +
            "Captures shared app audio while you use Teams.\n\n" +
            "Tap Start, approve audio capture, then switch to Teams."
        );
        title.setTextSize(18);
        title.setPadding(40, 60, 40, 40);

        Button start = new Button(this);
        start.setText("START CLASS CAPTURE");

        android.widget.LinearLayout layout =
            new android.widget.LinearLayout(this);

        layout.setOrientation(android.widget.LinearLayout.VERTICAL);
        layout.setPadding(30, 30, 30, 30);
        layout.addView(title);
        layout.addView(start);

        setContentView(layout);

        start.setOnClickListener(v -> startCapture());
    }

    private void startCapture() {

        MediaProjectionManager manager =
            (MediaProjectionManager)
                getSystemService(MEDIA_PROJECTION_SERVICE);

        Intent intent =
            manager.createScreenCaptureIntent();

        startActivityForResult(
            intent,
            REQUEST_CAPTURE
        );
    }

    @Override
    protected void onActivityResult(
        int requestCode,
        int resultCode,
        Intent data
    ) {
        super.onActivityResult(
            requestCode,
            resultCode,
            data
        );

        if(
            requestCode == REQUEST_CAPTURE &&
            resultCode == RESULT_OK &&
            data != null
        ) {

            Intent service =
                new Intent(
                    this,
                    CaptureService.class
                );

            service.putExtra(
                "resultCode",
                resultCode
            );

            service.putExtra(
                "data",
                data
            );

            startForegroundService(service);
        }
    }
}
