import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
import os

class Drawer:
    @staticmethod
    def draw_traffic_lines(frame, lines):
        # Kolory linii zewnętrznych (Outer)
        colors = {'up': (0,255,255), 'down': (0,165,255), 'left': (255,0,0), 'right': (42,42,165)}
        # Kolory linii wewnętrznych (Inner)
        in_colors = {'up': (0,0,255), 'down': (255,255,0), 'left': (0,255,0), 'right': (200,222,245)}
        
        for d in lines:
            if d in ['up', 'down']:
                cv2.line(frame, (0, lines[d]['outer']), (frame.shape[1], lines[d]['outer']), colors[d], 2)
                cv2.line(frame, (0, lines[d]['inner']), (frame.shape[1], lines[d]['inner']), in_colors[d], 2)
            else:
                cv2.line(frame, (lines[d]['outer'], 0), (lines[d]['outer'], frame.shape[0]), colors[d], 2)
                cv2.line(frame, (lines[d]['inner'], 0), (lines[d]['inner'], frame.shape[0]), in_colors[d], 2)

    @staticmethod
    def draw_ui(frame, counts):
        """Wyświetla statystyki w lewym górnym rogu"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 160), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_off = 25
        for d, types in counts.items():
            total = sum(types.values())
            text = f"{d}: {total} (O:{types['OSOBOWY']} B:{types['BUS']} C:{types['CIEZAROWY']} J:{types['JEDNOSLAD']})"
            cv2.putText(frame, text, (10, y_off), 1, 0.9, (255, 255, 255), 1)
            y_off += 30

    @staticmethod
    def save_summary_plot(df, background_frame, output_path):
        """Generuje obraz ze wszystkimi ścieżkami pojazdów na tle pierwszej klatki"""
        plt.figure(figsize=(12, 7))
        plt.imshow(cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB))
        
        for col in df.columns:
            points = df[col][df[col] != ""].tolist()
            if len(points) > 5:
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                
                color = [random.random() for _ in range(3)]
                plt.plot(x, y, linewidth=1.5, color=color, alpha=0.7)
                plt.scatter(x[0], y[0], color=color, s=10)

        plt.title("Podsumowanie ruchu - Trasy pojazdów")
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()