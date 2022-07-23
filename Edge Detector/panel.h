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
	void openHandler();
	void setThresholdValue(int);
	void setBrightnessValue(int);
	void compute();

private:
	Ui::PanelClass ui;
	EdgeDetector detector;
	bool isOpened;
	bool isProcessed;
	bool recentSaved;


	void resetSliders();
	void openFileChooser();
	void disableAll();
	void enableAll();
};