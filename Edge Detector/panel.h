#pragma once

#include <QMainWindow>
#include "ui_panel.h"
#include "EdgeDetector.h"


class Panel : public QMainWindow
{
	Q_OBJECT

public:
	Panel(QWidget *parent = nullptr);
	~Panel();

public slots:
	void handleOpen();
	void showEdgesMaximized();
	void handleClose();
	void handleSave();
	void handleExit();
	void setThresholdValue(int);
	void setBrightnessValue(int);
	void compute();

private:
	Ui::PanelClass ui;
	EdgeDetector detector;
	bool isOpened;
	bool isProcessed;
	bool isRecentlySaved;


	void resetSliders();
	void openFileChooser();
	void resetLabels();
};