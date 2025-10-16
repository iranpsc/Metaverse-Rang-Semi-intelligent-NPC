using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using System.Text;
using System.Threading;

public class UnityWebSocketClient : MonoBehaviour
{
    [Header("WebSocket Configuration")]
    public string serverUrl = "ws://localhost:8000/ws/audio-transcription/";
    
    [Header("Audio Configuration")]
    public int sampleRate = 16000;
    public int channels = 1;
    public int chunkSize = 1024;
    
    [Header("Debug")]
    public bool enableDebugLogs = true;
    
    private WebSocket webSocket;
    private AudioClip recordingClip;
    private bool isRecording = false;
    private bool isConnected = false;
    private int recordingPosition = 0;
    private float[] audioBuffer;
    private Thread audioThread;
    
    // Events for Unity
    public System.Action<string> OnTranscriptionReceived;
    public System.Action<string> OnCompleteTranscription;
    public System.Action<string> OnError;
    public System.Action OnConnected;
    public System.Action OnDisconnected;
    
    void Start()
    {
        ConnectToServer();
    }
    
    void OnDestroy()
    {
        DisconnectFromServer();
    }
    
    public void ConnectToServer()
    {
        if (webSocket != null)
        {
            webSocket.Close();
        }
        
        webSocket = new WebSocket(serverUrl);
        
        webSocket.OnOpen += (sender, e) =>
        {
            isConnected = true;
            if (enableDebugLogs) Debug.Log("WebSocket Connected");
            OnConnected?.Invoke();
        };
        
        webSocket.OnMessage += (sender, e) =>
        {
            if (e.IsBinary)
            {
                // Handle binary audio data (if needed)
                ProcessAudioData(e.RawData);
            }
            else
            {
                // Handle text messages
                ProcessTextMessage(e.Data);
            }
        };
        
        webSocket.OnError += (sender, e) =>
        {
            if (enableDebugLogs) Debug.LogError($"WebSocket Error: {e.Message}");
            OnError?.Invoke(e.Message);
        };
        
        webSocket.OnClose += (sender, e) =>
        {
            isConnected = false;
            if (enableDebugLogs) Debug.Log("WebSocket Disconnected");
            OnDisconnected?.Invoke();
        };
        
        webSocket.Connect();
    }
    
    public void DisconnectFromServer()
    {
        if (webSocket != null)
        {
            StopRecording();
            webSocket.Close();
            webSocket = null;
        }
    }
    
    private void ProcessTextMessage(string message)
    {
        try
        {
            var data = JsonUtility.FromJson<WebSocketMessage>(message);
            
            switch (data.type)
            {
                case "connection_established":
                    if (enableDebugLogs) Debug.Log($"Server: {data.message}");
                    break;
                    
                case "model_loaded":
                    if (enableDebugLogs) Debug.Log($"Server: {data.message}");
                    break;
                    
                case "transcription":
                    if (enableDebugLogs) Debug.Log($"[{data.timestamp:F2}s] Sentence {data.sentence_index + 1}/{data.total_sentences}: {data.text}");
                    OnTranscriptionReceived?.Invoke(data.text);
                    break;
                    
                case "complete_transcription":
                    if (enableDebugLogs) Debug.Log($"Complete Transcription: {data.text}");
                    OnCompleteTranscription?.Invoke(data.text);
                    break;
                    
                case "error":
                    if (enableDebugLogs) Debug.LogError($"Server Error: {data.message}");
                    OnError?.Invoke(data.message);
                    break;
                    
                case "recording_started":
                    if (enableDebugLogs) Debug.Log($"Server: {data.message}");
                    break;
                    
                case "recording_stopped":
                    if (enableDebugLogs) Debug.Log($"Server: {data.message}");
                    break;
            }
        }
        catch (Exception e)
        {
            if (enableDebugLogs) Debug.LogError($"Error processing message: {e.Message}");
        }
    }
    
    private void ProcessAudioData(byte[] audioData)
    {
        // Handle incoming audio data if needed
    }
    
    public void StartRecording()
    {
        if (!isConnected)
        {
            Debug.LogError("Not connected to server");
            return;
        }
        
        if (isRecording)
        {
            Debug.LogWarning("Already recording");
            return;
        }
        
        // Initialize audio recording
        recordingClip = Microphone.Start(null, true, 10, sampleRate);
        audioBuffer = new float[chunkSize];
        recordingPosition = 0;
        isRecording = true;
        
        // Send start command
        SendCommand("start_recording");
        
        // Start audio processing thread
        audioThread = new Thread(ProcessAudioThread);
        audioThread.Start();
        
        if (enableDebugLogs) Debug.Log("Recording started");
    }
    
    public void StopRecording()
    {
        if (!isRecording)
        {
            return;
        }
        
        isRecording = false;
        
        // Stop microphone
        if (Microphone.IsRecording(null))
        {
            Microphone.End(null);
        }
        
        // Stop audio thread
        if (audioThread != null && audioThread.IsAlive)
        {
            audioThread.Join();
        }
        
        // Send stop command
        SendCommand("stop_recording");
        
        if (enableDebugLogs) Debug.Log("Recording stopped");
    }
    
    private void ProcessAudioThread()
    {
        while (isRecording)
        {
            if (Microphone.IsRecording(null))
            {
                int position = Microphone.GetPosition(null);
                if (position < recordingPosition)
                {
                    // Handle wraparound
                    int samplesToRead = (recordingClip.samples - recordingPosition) + position;
                    float[] tempBuffer = new float[samplesToRead];
                    recordingClip.GetData(tempBuffer, recordingPosition);
                    
                    // Send audio data
                    SendAudioData(ConvertToBytes(tempBuffer));
                }
                else if (position > recordingPosition)
                {
                    // Normal case
                    int samplesToRead = position - recordingPosition;
                    if (samplesToRead > 0)
                    {
                        float[] tempBuffer = new float[samplesToRead];
                        recordingClip.GetData(tempBuffer, recordingPosition);
                        
                        // Send audio data
                        SendAudioData(ConvertToBytes(tempBuffer));
                    }
                }
                
                recordingPosition = position;
            }
            
            Thread.Sleep(10); // Small delay to prevent overwhelming the server
        }
    }
    
    private byte[] ConvertToBytes(float[] audioData)
    {
        byte[] bytes = new byte[audioData.Length * 2]; // 16-bit audio
        for (int i = 0; i < audioData.Length; i++)
        {
            short sample = (short)(audioData[i] * short.MaxValue);
            bytes[i * 2] = (byte)(sample & 0xFF);
            bytes[i * 2 + 1] = (byte)((sample >> 8) & 0xFF);
        }
        return bytes;
    }
    
    private void SendAudioData(byte[] audioData)
    {
        if (webSocket != null && isConnected)
        {
            webSocket.Send(audioData);
        }
    }
    
    private void SendCommand(string command, Dictionary<string, object> parameters = null)
    {
        if (webSocket != null && isConnected)
        {
            var message = new Dictionary<string, object>
            {
                ["command"] = command
            };
            
            if (parameters != null)
            {
                foreach (var param in parameters)
                {
                    message[param.Key] = param.Value;
                }
            }
            
            string json = JsonUtility.ToJson(new SerializableDictionary(message));
            webSocket.Send(json);
        }
    }
    
    public void SendAudioFile(string filePath)
    {
        if (!isConnected)
        {
            Debug.LogError("Not connected to server");
            return;
        }
        
        StartCoroutine(LoadAndSendAudioFile(filePath));
    }
    
    private IEnumerator LoadAndSendAudioFile(string filePath)
    {
        // Load audio file using Unity's WWW or UnityWebRequest
        using (var www = new WWW("file://" + filePath))
        {
            yield return www;
            
            if (www.error != null)
            {
                Debug.LogError($"Error loading audio file: {www.error}");
                yield break;
            }
            
            AudioClip clip = www.GetAudioClip();
            
            // Convert to bytes and send
            float[] samples = new float[clip.samples * clip.channels];
            clip.GetData(samples, 0);
            
            byte[] audioBytes = ConvertToBytes(samples);
            
            SendCommand("start_recording");
            
            // Send audio data in chunks
            int chunkSize = 1024;
            for (int i = 0; i < audioBytes.Length; i += chunkSize)
            {
                int remainingBytes = audioBytes.Length - i;
                int currentChunkSize = Mathf.Min(chunkSize, remainingBytes);
                
                byte[] chunk = new byte[currentChunkSize];
                Array.Copy(audioBytes, i, chunk, 0, currentChunkSize);
                
                SendAudioData(chunk);
                
                yield return new WaitForSeconds(0.01f); // Small delay
            }
            
            SendCommand("stop_recording");
        }
    }
    
    // Helper classes for JSON serialization
    [System.Serializable]
    public class WebSocketMessage
    {
        public string type;
        public string message;
        public string text;
        public int sentence_index;
        public int total_sentences;
        public float timestamp;
        public int buffer_size;
        public bool model_loaded;
    }
    
    [System.Serializable]
    public class SerializableDictionary
    {
        public Dictionary<string, object> data;
        
        public SerializableDictionary(Dictionary<string, object> dict)
        {
            data = dict;
        }
    }
}
