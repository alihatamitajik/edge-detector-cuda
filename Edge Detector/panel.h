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
	void open_file_chooser();

private:
	Ui::PanelClass ui;
	EdgeDetector detector;
	bool isOpened;
	bool isProcessed;
};