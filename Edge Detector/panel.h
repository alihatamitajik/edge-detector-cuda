#pragma once

#include <QMainWindow>
#include "ui_panel.h"

class Panel : public QMainWindow
{
	Q_OBJECT

public:
	Panel(QWidget *parent = nullptr);
	~Panel();

private:
	Ui::PanelClass ui;
};
